#!/usr/bin/env python3
"""
Fast Expert Node — binary transport + persistent connections.
Drop-in replacement for expert_node.py with /compute_bin binary endpoint.
"""

import argparse
import base64
import json
import os
import struct
import time
import gc

import numpy as np

PAGE_SIZE = 16384
BITS = 4
GROUP_SIZE = 64


def parse_layer_header(layer_path):
    with open(layer_path, "rb") as f:
        raw = f.read(PAGE_SIZE)
    return json.loads(raw.rstrip(b"\x00"))


def load_expert_from_bin(fd, expert_id, layout, tensor_layout):
    import mlx.core as mx
    MLX_DTYPES = {
        "uint32": mx.uint32, "float16": mx.float16,
        "float32": mx.float32, "bfloat16": mx.bfloat16,
    }
    data_start = layout["data_start"]
    block_size = layout["expert_block_size"]
    offset = data_start + expert_id * block_size
    raw = os.pread(fd, block_size, offset)
    result = {}
    for name, info in tensor_layout.items():
        inner_offset = info["inner_offset"]
        nbytes = info["nbytes"]
        shape = info["shape_per_expert"]
        dtype_str = info["dtype"].replace("mlx.core.", "")
        mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)
        arr_bytes = raw[inner_offset:inner_offset + nbytes]
        flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
        arr = flat.view(mlx_dtype).reshape(shape)
        result[name] = arr
    return result


def load_partition(model_dir, partition_start, partition_end, num_layers):
    import mlx.core as mx
    expert_dir = os.path.join(model_dir, "bin")
    experts = {}
    total_bytes = 0
    for layer_idx in range(num_layers):
        layer_path = os.path.join(expert_dir, f"layer_{layer_idx:02d}.bin")
        header = parse_layer_header(layer_path)
        layout = header["layout"]
        tensor_layout = layout["tensors"]
        fd = os.open(layer_path, os.O_RDONLY)
        try:
            for eid in range(partition_start, partition_end + 1):
                parsed = load_expert_from_bin(fd, eid, layout, tensor_layout)
                experts[(layer_idx, eid)] = parsed
                total_bytes += layout["expert_block_size"]
        finally:
            os.close(fd)
        all_arrays = []
        for eid in range(partition_start, partition_end + 1):
            all_arrays.extend(experts[(layer_idx, eid)].values())
        mx.eval(*all_arrays)
        n = partition_end - partition_start + 1
        print(f"  Layer {layer_idx:2d}: {n} experts loaded")
    return experts, total_bytes


def compute_expert_ffn(x, expert_data_list, local_indices, top_k_weights):
    """gather_qmm SwiGLU FFN for experts in this partition."""
    import mlx.core as mx
    import mlx.nn as nn
    if not expert_data_list:
        return mx.zeros_like(x)
    active_ids = [eid for eid, _ in expert_data_list]
    data_map = {eid: d for eid, d in expert_data_list}

    def stack_proj(proj):
        w = mx.stack([data_map[eid][f"switch_mlp.{proj}.weight"] for eid in active_ids])
        s = mx.stack([data_map[eid][f"switch_mlp.{proj}.scales"] for eid in active_ids])
        b = mx.stack([data_map[eid][f"switch_mlp.{proj}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    x_exp = mx.expand_dims(x, (-2, -3))
    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
                              rhs_indices=local_indices, transpose=True,
                              group_size=GROUP_SIZE, bits=BITS)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
                            rhs_indices=local_indices, transpose=True,
                            group_size=GROUP_SIZE, bits=BITS)
    hidden = nn.silu(gate_out) * up_out
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

    app = FastAPI(title="Fast Expert Sniper Node",
                  description=f"Partition [{partition_start}-{partition_end}]")
    partition_set = set(range(partition_start, partition_end + 1))
    stats = {"count": 0, "time": 0.0}

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
    async def compute(req: ComputeRequest):
        """JSON endpoint (backward compatible)."""
        t0 = time.time()
        h_np = np.frombuffer(base64.b64decode(req.hidden_state), dtype=np.float16)
        h_np = h_np.reshape(req.hidden_state_shape)
        x = mx.array(h_np)

        inds_np = np.frombuffer(base64.b64decode(req.top_k_indices), dtype=np.int32)
        inds_np = inds_np.reshape(req.top_k_indices_shape)

        weights_np = np.frombuffer(base64.b64decode(req.top_k_weights), dtype=np.float32)
        weights_np = weights_np.reshape(req.top_k_weights_shape)
        top_k_weights_mx = mx.array(weights_np)

        my_ids = sorted(eid for eid in req.expert_ids if eid in partition_set)

        if not my_ids:
            out_np = np.zeros(req.hidden_state_shape, dtype=np.float16)
            elapsed = time.time() - t0
            stats["count"] += 1; stats["time"] += elapsed
            return JSONResponse({
                "result": base64.b64encode(out_np.tobytes()).decode(),
                "shape": req.hidden_state_shape,
                "num_experts_computed": 0,
                "time_ms": elapsed * 1000,
            })

        id_to_local = {eid: i for i, eid in enumerate(my_ids)}
        local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
        local_indices = mx.array(local_np)
        mask_np = np.vectorize(lambda v: 1.0 if int(v) in partition_set else 0.0)(inds_np)
        masked_weights = top_k_weights_mx * mx.array(mask_np.astype(np.float32))

        expert_data_list = [(eid, experts[(req.layer_idx, eid)])
                            for eid in my_ids if (req.layer_idx, eid) in experts]

        if not expert_data_list:
            out_np = np.zeros(req.hidden_state_shape, dtype=np.float16)
            elapsed = time.time() - t0
            stats["count"] += 1; stats["time"] += elapsed
            return JSONResponse({
                "result": base64.b64encode(out_np.tobytes()).decode(),
                "shape": req.hidden_state_shape,
                "num_experts_computed": 0,
                "time_ms": elapsed * 1000,
            })

        result = compute_expert_ffn(x, expert_data_list, local_indices, masked_weights)
        mx.eval(result)
        out_np = np.array(result.astype(mx.float16))
        elapsed = time.time() - t0
        stats["count"] += 1; stats["time"] += elapsed
        return JSONResponse({
            "result": base64.b64encode(out_np.tobytes()).decode(),
            "shape": list(out_np.shape),
            "num_experts_computed": len(expert_data_list),
            "time_ms": elapsed * 1000,
        })

    @app.post("/compute_bin")
    async def compute_bin(request: Request):
        """Binary endpoint — no JSON/base64 overhead."""
        t0 = time.time()
        raw = await request.body()

        # Parse header: [layer_idx:u16][n_experts:u16]
        layer_idx, n_experts = struct.unpack("<HH", raw[:4])
        off = 4

        # Expert IDs
        expert_ids_arr = np.frombuffer(raw[off:off+n_experts*2], dtype=np.uint16)
        expert_ids_list = [int(e) for e in expert_ids_arr]
        off += n_experts * 2

        # Shapes (3 x uint16 each)
        hs_shape = tuple(int(x) for x in np.frombuffer(raw[off:off+6], dtype=np.uint16))
        off += 6
        inds_shape = tuple(int(x) for x in np.frombuffer(raw[off:off+6], dtype=np.uint16))
        off += 6
        wt_shape = tuple(int(x) for x in np.frombuffer(raw[off:off+6], dtype=np.uint16))
        off += 6

        # Tensors
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

        if not my_ids:
            elapsed = time.time() - t0
            stats["count"] += 1; stats["time"] += elapsed
            return make_empty_response()

        id_to_local = {eid: i for i, eid in enumerate(my_ids)}
        local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
        local_indices = mx.array(local_np)
        mask_np = np.vectorize(lambda v: 1.0 if int(v) in partition_set else 0.0)(inds_np)
        masked_weights = top_k_weights_mx * mx.array(mask_np.astype(np.float32))

        expert_data_list = [(eid, experts[(layer_idx, eid)])
                            for eid in my_ids if (layer_idx, eid) in experts]

        if not expert_data_list:
            elapsed = time.time() - t0
            stats["count"] += 1; stats["time"] += elapsed
            return make_empty_response()

        result = compute_expert_ffn(x, expert_data_list, local_indices, masked_weights)
        mx.eval(result)

        out_np = np.array(result.astype(mx.float16))
        elapsed = time.time() - t0
        stats["count"] += 1; stats["time"] += elapsed

        n_computed = len(expert_data_list)
        resp_hdr = struct.pack("<HH", n_computed, len(out_np.shape))
        resp_shape = np.array(out_np.shape, dtype=np.uint16).tobytes()
        return Response(content=resp_hdr + resp_shape + out_np.tobytes(),
                       media_type="application/octet-stream")

    @app.get("/health")
    async def health():
        n = partition_end - partition_start + 1
        avg_ms = (stats["time"] / stats["count"] * 1000) if stats["count"] > 0 else 0
        return {
            "status": "ok",
            "partition": f"{partition_start}-{partition_end}",
            "experts_per_layer": n,
            "total_experts_loaded": len(experts),
            "num_layers": num_layers,
            "memory_gb": round(mx.get_active_memory() / 1e9, 2),
            "compute_requests": stats["count"],
            "avg_compute_ms": round(avg_ms, 2),
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Fast Expert Sniper Node")
    parser.add_argument("--partition", required=True, help="e.g. '0-127'")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8301)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--memory-limit-gb", type=float, default=14.0)
    args = parser.parse_args()

    parts = args.partition.split("-")
    p_start, p_end = int(parts[0]), int(parts[1])

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_experts"]
    assert 0 <= p_start <= p_end < num_experts

    import mlx.core as mx
    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    mx.set_cache_limit(512 * 1024**2)

    n = p_end - p_start + 1
    est_gb = n * num_layers * 1769472 / 1e9
    print(f"Fast Expert Sniper Node [{p_start}-{p_end}]")
    print(f"  {n} experts/layer x {num_layers} layers = {est_gb:.1f} GB est.")
    print(f"  Port: {args.port}\n")

    t0 = time.time()
    experts, total_bytes = load_partition(args.model_dir, p_start, p_end, num_layers)
    gc.collect(); mx.clear_cache()
    print(f"\nLoaded {len(experts)} blocks ({total_bytes/1e9:.1f} GB) "
          f"in {time.time()-t0:.1f}s")
    print(f"Active memory: {mx.get_active_memory()/1e9:.2f} GB\n")

    app = create_app(experts, p_start, p_end, num_layers)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
