#!/usr/bin/env python3
"""
Gemma 4 Expert Node — loads partition of stacked SwitchLinear experts.

Gemma 4 stores experts as SwitchLinear tensors with shape (128, out, in).
Each node loads the full tensors then slices its partition (e.g., experts 0-63).

Usage:
    python gemma4_expert_node.py --partition 0-63 --model-dir ~/models/gemma4-4bit --port 8401
"""

import argparse
import base64
import json
import os
import struct
import time
import gc
import glob

import numpy as np


def load_partition(model_dir, partition_start, partition_end, num_layers):
    """Load expert partition by slicing SwitchLinear stacked tensors."""
    import mlx.core as mx

    print(f"Loading safetensors files...")
    all_weights = {}
    for sf in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        print(f"  {os.path.basename(sf)}")
        all_weights.update(mx.load(sf))

    experts = {}  # (layer_idx, tensor_name) → sliced tensor
    total_bytes = 0

    for layer_idx in range(num_layers):
        prefix = f"language_model.model.layers.{layer_idx}.experts.switch_glu"
        layer_tensors = {}

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for comp in ["weight", "scales", "biases"]:
                key = f"{prefix}.{proj}.{comp}"
                if key in all_weights:
                    full = all_weights[key]  # (128, out, in)
                    sliced = full[partition_start:partition_end + 1]  # (N, out, in)
                    layer_tensors[f"switch_glu.{proj}.{comp}"] = sliced
                    total_bytes += sliced.nbytes

        if layer_tensors:
            mx.eval(*layer_tensors.values())
            experts[layer_idx] = layer_tensors
            n = partition_end - partition_start + 1
            size_mb = sum(t.nbytes for t in layer_tensors.values()) / 1e6
            print(f"  Layer {layer_idx:2d}: {n} experts, {size_mb:.1f} MB")

    # Free original weights
    del all_weights
    gc.collect()
    mx.clear_cache()

    return experts, total_bytes


def compute_expert_ffn(x, layer_tensors, local_indices, top_k_weights, partition_start):
    """gather_qmm SwiGLU FFN for experts in this partition.
    
    Gemma 4 uses 8-bit quantized experts (group_size=64, bits=8).
    """
    import mlx.core as mx
    import mlx.nn as nn

    BITS = 4
    GROUP_SIZE = 64

    gate_w = layer_tensors["switch_glu.gate_proj.weight"]
    gate_s = layer_tensors["switch_glu.gate_proj.scales"]
    gate_b = layer_tensors["switch_glu.gate_proj.biases"]
    up_w = layer_tensors["switch_glu.up_proj.weight"]
    up_s = layer_tensors["switch_glu.up_proj.scales"]
    up_b = layer_tensors["switch_glu.up_proj.biases"]
    down_w = layer_tensors["switch_glu.down_proj.weight"]
    down_s = layer_tensors["switch_glu.down_proj.scales"]
    down_b = layer_tensors["switch_glu.down_proj.biases"]

    x_exp = mx.expand_dims(x, (-2, -3))

    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
                              rhs_indices=local_indices, transpose=True,
                              group_size=GROUP_SIZE, bits=BITS)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
                            rhs_indices=local_indices, transpose=True,
                            group_size=GROUP_SIZE, bits=BITS)

    # Gemma 4 uses gelu, not silu
    hidden = nn.gelu_approx(gate_out) * up_out

    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
                              rhs_indices=local_indices, transpose=True,
                              group_size=GROUP_SIZE, bits=BITS)

    out = down_out.squeeze(-2)
    out = (out * top_k_weights[..., None]).sum(axis=-2)
    return out


def create_app(experts, partition_start, partition_end, num_layers):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel
    from typing import List
    import mlx.core as mx

    app = FastAPI(title="Gemma 4 Expert Node",
                  description=f"Partition [{partition_start}-{partition_end}]")
    partition_set = set(range(partition_start, partition_end + 1))
    stats = {"count": 0, "time": 0.0}

    @app.post("/compute_bin")
    async def compute_bin(request: Request):
        """Binary endpoint for expert FFN computation."""
        t0 = time.time()
        raw = await request.body()

        layer_idx, n_experts = struct.unpack("<HH", raw[:4])
        off = 4

        expert_ids_arr = np.frombuffer(raw[off:off+n_experts*2], dtype=np.uint16)
        expert_ids_list = [int(e) for e in expert_ids_arr]
        off += n_experts * 2

        hs_shape = tuple(int(x) for x in np.frombuffer(raw[off:off+6], dtype=np.uint16))
        off += 6
        inds_shape = tuple(int(x) for x in np.frombuffer(raw[off:off+6], dtype=np.uint16))
        off += 6
        wt_shape = tuple(int(x) for x in np.frombuffer(raw[off:off+6], dtype=np.uint16))
        off += 6

        hs_bytes = int(np.prod(hs_shape)) * 2
        h_np = np.frombuffer(raw[off:off+hs_bytes], dtype=np.float16).copy().reshape(hs_shape)
        x = mx.array(h_np)
        off += hs_bytes

        inds_bytes = int(np.prod(inds_shape)) * 4
        inds_np = np.frombuffer(raw[off:off+inds_bytes], dtype=np.int32).copy().reshape(inds_shape)
        off += inds_bytes

        wt_bytes = int(np.prod(wt_shape)) * 4
        weights_np = np.frombuffer(raw[off:off+wt_bytes], dtype=np.float32).copy().reshape(wt_shape)
        top_k_weights_mx = mx.array(weights_np)

        my_ids = sorted(eid for eid in expert_ids_list if eid in partition_set)

        def make_empty_response():
            out_np = np.zeros(hs_shape, dtype=np.float16)
            resp_hdr = struct.pack("<HH", 0, len(hs_shape))
            resp_shape = np.array(hs_shape, dtype=np.uint16).tobytes()
            return Response(content=resp_hdr + resp_shape + out_np.tobytes(),
                          media_type="application/octet-stream")

        if not my_ids or layer_idx not in experts:
            elapsed = time.time() - t0
            stats["count"] += 1; stats["time"] += elapsed
            return make_empty_response()

        # Map global expert IDs to local indices (offset by partition_start)
        # local_indices maps top-k slots to indices within our stacked tensor
        id_to_local = {eid: eid - partition_start for eid in my_ids}
        local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
        local_indices = mx.array(local_np)
        mask_np = np.vectorize(lambda v: 1.0 if int(v) in partition_set else 0.0)(inds_np)
        masked_weights = top_k_weights_mx * mx.array(mask_np.astype(np.float32))

        layer_tensors = experts[layer_idx]
        result = compute_expert_ffn(x, layer_tensors, local_indices, masked_weights, partition_start)
        mx.eval(result)

        out_np = np.array(result.astype(mx.float16))
        elapsed = time.time() - t0
        stats["count"] += 1; stats["time"] += elapsed

        n_computed = len(my_ids)
        resp_hdr = struct.pack("<HH", n_computed, len(out_np.shape))
        resp_shape = np.array(out_np.shape, dtype=np.uint16).tobytes()
        return Response(content=resp_hdr + resp_shape + out_np.tobytes(),
                       media_type="application/octet-stream")

    # Also keep JSON endpoint for compatibility
    class ComputeRequest(BaseModel):
        layer_idx: int
        expert_ids: List[int]
        top_k_indices: str
        top_k_indices_shape: List[int]
        hidden_state: str
        hidden_state_shape: List[int]
        top_k_weights: str
        top_k_weights_shape: List[int]

    @app.post("/compute")
    async def compute_json(req: ComputeRequest):
        t0 = time.time()
        h_np = np.frombuffer(base64.b64decode(req.hidden_state), dtype=np.float16).reshape(req.hidden_state_shape)
        x = mx.array(h_np)
        inds_np = np.frombuffer(base64.b64decode(req.top_k_indices), dtype=np.int32).reshape(req.top_k_indices_shape)
        weights_np = np.frombuffer(base64.b64decode(req.top_k_weights), dtype=np.float32).reshape(req.top_k_weights_shape)
        top_k_weights_mx = mx.array(weights_np)

        my_ids = sorted(eid for eid in req.expert_ids if eid in partition_set)

        if not my_ids or req.layer_idx not in experts:
            out_np = np.zeros(req.hidden_state_shape, dtype=np.float16)
            elapsed = time.time() - t0
            stats["count"] += 1; stats["time"] += elapsed
            return JSONResponse({"result": base64.b64encode(out_np.tobytes()).decode(),
                               "shape": req.hidden_state_shape, "num_experts_computed": 0,
                               "time_ms": elapsed * 1000})

        id_to_local = {eid: eid - partition_start for eid in my_ids}
        local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
        local_indices = mx.array(local_np)
        mask_np = np.vectorize(lambda v: 1.0 if int(v) in partition_set else 0.0)(inds_np)
        masked_weights = top_k_weights_mx * mx.array(mask_np.astype(np.float32))

        result = compute_expert_ffn(x, experts[req.layer_idx], local_indices, masked_weights, partition_start)
        mx.eval(result)
        out_np = np.array(result.astype(mx.float16))
        elapsed = time.time() - t0
        stats["count"] += 1; stats["time"] += elapsed
        return JSONResponse({"result": base64.b64encode(out_np.tobytes()).decode(),
                           "shape": list(out_np.shape), "num_experts_computed": len(my_ids),
                           "time_ms": elapsed * 1000})

    @app.get("/health")
    async def health():
        n = partition_end - partition_start + 1
        avg_ms = (stats["time"] / stats["count"] * 1000) if stats["count"] > 0 else 0
        return {
            "status": "ok",
            "model": "gemma4-26b-a4b",
            "partition": f"{partition_start}-{partition_end}",
            "experts_per_layer": n,
            "total_layers_loaded": len(experts),
            "num_layers": num_layers,
            "memory_gb": round(mx.get_active_memory() / 1e9, 2),
            "compute_requests": stats["count"],
            "avg_compute_ms": round(avg_ms, 2),
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Expert Sniper Node")
    parser.add_argument("--partition", required=True, help="e.g. '0-63'")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8401)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--memory-limit-gb", type=float, default=14.0)
    args = parser.parse_args()

    parts = args.partition.split("-")
    p_start, p_end = int(parts[0]), int(parts[1])

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    num_layers = text_config["num_hidden_layers"]
    num_experts = text_config["num_experts"]
    assert 0 <= p_start <= p_end < num_experts

    import mlx.core as mx
    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    mx.set_cache_limit(512 * 1024**2)

    n = p_end - p_start + 1
    print(f"Gemma 4 Expert Sniper Node [{p_start}-{p_end}]")
    print(f"  {n} experts/layer x {num_layers} layers")
    print(f"  Port: {args.port}\n")

    t0 = time.time()
    expert_data, total_bytes = load_partition(args.model_dir, p_start, p_end, num_layers)
    gc.collect(); mx.clear_cache()
    print(f"\nLoaded {len(expert_data)} layers ({total_bytes/1e9:.1f} GB) "
          f"in {time.time()-t0:.1f}s")
    print(f"Active memory: {mx.get_active_memory()/1e9:.2f} GB\n")

    app = create_app(expert_data, p_start, p_end, num_layers)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
