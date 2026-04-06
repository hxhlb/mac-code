#!/usr/bin/env python3
"""
Split Gemma 4 MLX model into expert layer files + pinned weights.

Takes the mlx-community/gemma-4-26b-a4b-it-4bit safetensors and creates:
  - pinned.safetensors: attention, embeddings, norms, shared layers (~2-3 GB)
  - bin/layer_XX.bin: expert weights per layer (~400 MB each, 30 files)

Usage:
  python split_gemma4.py \
    --input ~/models/gemma4-26b-4bit \
    --output ~/models/gemma4-26b-stream
"""

import argparse
import json
import os
import struct
import time

import numpy as np
import mlx.core as mx


def load_all_weights(model_dir):
    """Load all safetensors files from model directory."""
    import glob
    weights = {}
    for sf in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        print(f"  Loading {os.path.basename(sf)}...")
        w = mx.load(sf)
        weights.update(w)
    return weights


def identify_expert_keys(weights):
    """Identify which keys are expert weights vs pinned weights.

    Gemma 4 expert keys follow pattern:
      text_model.model.layers.X.block_sparse_moe.experts.Y.{gate_proj,up_proj,down_proj}.{weight,scales,biases}

    Everything else is pinned (attention, norms, embeddings, router, etc).
    """
    expert_keys = {}   # (layer_idx, expert_idx) → {tensor_name: key}
    pinned_keys = {}   # key → True

    for key in weights:
        # Check if this is an expert weight
        # Pattern: ...layers.X.block_sparse_moe.experts.Y.proj.component
        parts = key.split(".")
        is_expert = False
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts):
                # Found expert weight
                layer_str = None
                for j in range(i):
                    if parts[j] == "layers" and j + 1 < len(parts):
                        layer_str = parts[j + 1]
                        break
                if layer_str is not None:
                    layer_idx = int(layer_str)
                    expert_idx = int(parts[i + 1])
                    # Tensor name: everything after expert_idx
                    tensor_name = ".".join(parts[i + 2:])
                    if (layer_idx, expert_idx) not in expert_keys:
                        expert_keys[(layer_idx, expert_idx)] = {}
                    expert_keys[(layer_idx, expert_idx)][tensor_name] = key
                    is_expert = True
                break

        if not is_expert:
            pinned_keys[key] = True

    return expert_keys, pinned_keys


def build_layer_files(weights, expert_keys, output_dir, num_layers, num_experts):
    """Create layer_XX.bin files with expert data."""

    bin_dir = os.path.join(output_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)

    PAGE_SIZE = 16384

    for layer_idx in range(num_layers):
        # Collect all experts for this layer
        layer_experts = {}
        for (l, e), tensors in expert_keys.items():
            if l == layer_idx:
                layer_experts[e] = tensors

        if not layer_experts:
            print(f"  Layer {layer_idx}: no experts found, skipping")
            continue

        # Determine tensor layout from first expert
        first_expert = layer_experts[min(layer_experts.keys())]
        tensor_layout = {}
        inner_offset = 0

        for tensor_name in sorted(first_expert.keys()):
            key = first_expert[tensor_name]
            arr = weights[key]
            nbytes = arr.nbytes
            shape = list(arr.shape)
            dtype = str(arr.dtype).replace("mlx.core.", "")

            tensor_layout[tensor_name] = {
                "inner_offset": inner_offset,
                "nbytes": nbytes,
                "shape_per_expert": shape,
                "dtype": dtype,
            }
            inner_offset += nbytes

        expert_block_size = inner_offset
        data_start = PAGE_SIZE

        # Build header
        header = {
            "format": "expert_sniper_v1",
            "model": "gemma4-26b-a4b",
            "layer_idx": layer_idx,
            "num_experts": num_experts,
            "layout": {
                "expert_block_size": expert_block_size,
                "data_start": data_start,
                "tensors": tensor_layout,
            }
        }

        header_bytes = json.dumps(header, indent=2).encode("utf-8")
        assert len(header_bytes) < PAGE_SIZE, f"Header too large: {len(header_bytes)} bytes"
        header_padded = header_bytes + b"\x00" * (PAGE_SIZE - len(header_bytes))

        # Write layer file
        layer_path = os.path.join(bin_dir, f"layer_{layer_idx:02d}.bin")
        with open(layer_path, "wb") as f:
            f.write(header_padded)

            for expert_idx in range(num_experts):
                if expert_idx in layer_experts:
                    tensors = layer_experts[expert_idx]
                    expert_data = bytearray()
                    for tensor_name in sorted(tensors.keys()):
                        key = tensors[tensor_name]
                        arr = weights[key]
                        # Convert to numpy for raw bytes
                        if arr.dtype == mx.uint32:
                            np_arr = np.array(arr).view(np.uint32)
                        elif arr.dtype == mx.bfloat16:
                            np_arr = np.array(arr.astype(mx.float16)).view(np.uint16)
                        elif arr.dtype == mx.float32:
                            np_arr = np.array(arr).view(np.float32)
                        elif arr.dtype == mx.float16:
                            np_arr = np.array(arr).view(np.float16)
                        else:
                            np_arr = np.array(arr)
                        expert_data.extend(np_arr.tobytes())

                    # Pad to exact block size
                    if len(expert_data) < expert_block_size:
                        expert_data.extend(b"\x00" * (expert_block_size - len(expert_data)))
                    f.write(bytes(expert_data[:expert_block_size]))
                else:
                    # Empty expert slot
                    f.write(b"\x00" * expert_block_size)

        file_size = os.path.getsize(layer_path)
        print(f"  Layer {layer_idx:2d}: {len(layer_experts)} experts, "
              f"block={expert_block_size} bytes, file={file_size/1e6:.1f} MB")

    return expert_block_size


def save_pinned(weights, pinned_keys, output_dir):
    """Save non-expert weights to pinned.safetensors."""
    pinned = {k: weights[k] for k in pinned_keys}

    pinned_path = os.path.join(output_dir, "pinned.safetensors")
    mx.save_safetensors(pinned_path, pinned)

    size = os.path.getsize(pinned_path) / 1e9
    print(f"  Pinned weights: {len(pinned)} tensors, {size:.1f} GB")
    return size


def main():
    parser = argparse.ArgumentParser(description="Split Gemma 4 for distributed Expert Sniper")
    parser.add_argument("--input", required=True, help="Input model directory")
    parser.add_argument("--output", required=True, help="Output directory for split model")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    config_path = os.path.join(args.input, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    num_layers = text_config["num_hidden_layers"]
    num_experts = text_config["num_experts"]

    print(f"Gemma 4 Split Tool")
    print(f"  Layers: {num_layers}, Experts: {num_experts}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print()

    # Load all weights
    print("Loading model weights...")
    t0 = time.time()
    weights = load_all_weights(args.input)
    print(f"  Loaded {len(weights)} tensors in {time.time()-t0:.1f}s")
    print()

    # Identify expert vs pinned keys
    expert_keys, pinned_keys = identify_expert_keys(weights)
    n_expert_tensors = sum(len(v) for v in expert_keys.values())
    print(f"Expert tensors: {n_expert_tensors} ({len(expert_keys)} expert slots)")
    print(f"Pinned tensors: {len(pinned_keys)}")
    print()

    # Save pinned weights
    print("Saving pinned weights...")
    save_pinned(weights, pinned_keys, args.output)
    print()

    # Build layer files
    print("Building expert layer files...")
    block_size = build_layer_files(weights, expert_keys, args.output, num_layers, num_experts)
    print()

    # Copy config and tokenizer files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "generation_config.json", "chat_template.jinja", "processor_config.json"]:
        src = os.path.join(args.input, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(args.output, fname))

    # Write sniper config
    sniper_config = {
        "version": 1,
        "model": "gemma4-26b-a4b",
        "model_dir": args.output,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": text_config.get("top_k_experts", 8),
        "expert_block_bytes": block_size,
        "hidden_size": text_config["hidden_size"],
        "moe_intermediate_size": text_config.get("moe_intermediate_size", 704),
    }
    with open(os.path.join(args.output, "sniper_config.json"), "w") as f:
        json.dump(sniper_config, f, indent=2)

    print(f"Split complete! Output at {args.output}")
    print(f"  pinned.safetensors — non-expert weights")
    print(f"  bin/layer_XX.bin — expert weights ({num_layers} files)")
    print(f"  sniper_config.json — sniper configuration")


if __name__ == "__main__":
    main()
