#!/usr/bin/env python3
"""Split Qwen3-30B-A3B-4bit into pinned (attn+embed) + streaming experts."""
import os, json, gc, time, struct
import numpy as np
import mlx.core as mx

MLX_MODEL_DIR = "/Users/bigneek/models/qwen3-30b-mlx-4bit"
OUTPUT_DIR = "/Users/bigneek/models/qwen3-30b-stream"
NUM_LAYERS = 48
NUM_EXPERTS = 128
GROUP_SIZE = 64
PAGE_SIZE = 16384

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/bin", exist_ok=True)

    # Load config
    config = json.load(open(f"{MLX_MODEL_DIR}/config.json"))

    # Load all shards
    import glob
    shard_files = sorted(glob.glob(f"{MLX_MODEL_DIR}/model-*.safetensors"))
    print(f"Loading {len(shard_files)} shards...")
    all_weights = {}
    for sf in shard_files:
        w = mx.load(sf)
        all_weights.update(w)
        print(f"  {os.path.basename(sf)}: {len(w)} keys")
    print(f"Total: {len(all_weights)} keys")

    # Separate pinned (non-expert) vs expert weights
    pinned = {}
    expert_keys = {}  # layer_idx -> list of (name, tensor)
    
    for k, v in all_weights.items():
        if "switch_mlp" in k:
            # Expert weight - parse layer index
            import re
            m = re.search(r"layers\.(\d+)\.", k)
            layer_idx = int(m.group(1))
            # Remove model.layers.N.mlp. prefix, keep switch_mlp.xxx
            local_name = k.split(f"layers.{layer_idx}.mlp.")[-1]
            if layer_idx not in expert_keys:
                expert_keys[layer_idx] = []
            expert_keys[layer_idx].append((local_name, v))
        else:
            pinned[k] = v

    print(f"Pinned: {len(pinned)} keys, {sum(v.nbytes for v in pinned.values())/1e9:.2f} GB")
    print(f"Expert layers: {len(expert_keys)}")

    # Save pinned
    mx.savez(f"{OUTPUT_DIR}/pinned.npz", **pinned)
    # Actually save as safetensors
    mx.save_safetensors(f"{OUTPUT_DIR}/pinned.safetensors", pinned)
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    print(f"Saved pinned.safetensors: {pinned_bytes/1e9:.2f} GB")
    del pinned
    gc.collect()

    # Process expert layers
    total_expert_bytes = 0
    t0 = time.time()

    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in expert_keys:
            print(f"  Layer {layer_idx}: no experts (dense MLP)")
            continue

        layer_tensors = dict(expert_keys[layer_idx])
        # Each tensor is (NUM_EXPERTS, ...) - we need to split per expert
        
        # Figure out per-expert layout
        # Tensor order: gate_proj.weight, gate_proj.scales, gate_proj.biases,
        #               up_proj.weight, up_proj.scales, up_proj.biases,
        #               down_proj.weight, down_proj.scales, down_proj.biases
        tensor_order = [
            "switch_mlp.gate_proj.weight", "switch_mlp.gate_proj.scales", "switch_mlp.gate_proj.biases",
            "switch_mlp.up_proj.weight", "switch_mlp.up_proj.scales", "switch_mlp.up_proj.biases",
            "switch_mlp.down_proj.weight", "switch_mlp.down_proj.scales", "switch_mlp.down_proj.biases",
        ]

        # Calculate per-expert sizes
        tensor_info = {}
        offset = 0
        for tname in tensor_order:
            t = layer_tensors[tname]
            per_expert_shape = list(t.shape[1:])  # remove expert dim
            per_expert_bytes = int(np.prod(per_expert_shape)) * t.dtype.size
            tensor_info[tname] = {
                "inner_offset": offset,
                "nbytes": per_expert_bytes,
                "shape_per_expert": per_expert_shape,
                "dtype": str(t.dtype),
            }
            offset += per_expert_bytes

        # Pad to page alignment
        expert_block_size = ((offset + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

        # Build header
        header = {
            "layer_idx": layer_idx,
            "num_experts": NUM_EXPERTS,
            "layout": {
                "expert_block_size": expert_block_size,
                "data_start": PAGE_SIZE,
                "tensors": tensor_info,
            }
        }
        header_json = json.dumps(header).encode()
        header_padded = header_json + b"\x00" * (PAGE_SIZE - len(header_json))

        # Write layer file
        layer_path = f"{OUTPUT_DIR}/bin/moe_layer_{layer_idx:02d}.bin"
        layer_bytes = 0
        with open(layer_path, "wb") as f:
            f.write(header_padded)
            layer_bytes += PAGE_SIZE

            for eid in range(NUM_EXPERTS):
                expert_data = bytearray()
                for tname in tensor_order:
                    t = layer_tensors[tname]
                    expert_t = t[eid]
                    mx.eval(expert_t)
                    # Convert to bytes
                    # Handle bfloat16: convert via uint16 view
                    if expert_t.dtype == mx.bfloat16:
                        raw = np.array(expert_t.view(mx.uint16)).tobytes()
                    else:
                        raw = np.array(expert_t).tobytes()
                    expert_data.extend(raw)
                
                # Pad to expert_block_size
                pad = expert_block_size - len(expert_data)
                if pad > 0:
                    expert_data.extend(b"\x00" * pad)
                f.write(bytes(expert_data))
                layer_bytes += expert_block_size

        total_expert_bytes += layer_bytes
        elapsed = time.time() - t0
        print(f"  Layer {layer_idx:2d}/{NUM_LAYERS}: {layer_bytes/1e6:.1f} MB ({elapsed:.0f}s)")

        # Free memory
        del layer_tensors
        if layer_idx in expert_keys:
            del expert_keys[layer_idx]
        gc.collect()

    # Also symlink layer_XX.bin -> moe_layer_XX.bin
    for i in range(NUM_LAYERS):
        src = f"moe_layer_{i:02d}.bin"
        dst = f"{OUTPUT_DIR}/bin/layer_{i:02d}.bin"
        if os.path.exists(f"{OUTPUT_DIR}/bin/{src}") and not os.path.exists(dst):
            os.symlink(src, dst)

    # Write streaming config
    stream_config = {
        "model_type": "qwen3_moe",
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "rms_norm_eps": config["rms_norm_eps"],
        "vocab_size": config["vocab_size"],
        "max_position_embeddings": config.get("max_position_embeddings", 40960),
        "head_dim": config.get("head_dim"),
        "tie_word_embeddings": config.get("tie_word_embeddings", True),
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": config["num_experts_per_tok"],
        "moe_intermediate_size": config["moe_intermediate_size"],
        "shared_expert_intermediate_size": config.get("shared_expert_intermediate_size"),
        "norm_topk_prob": config.get("norm_topk_prob", True),
        "mlp_only_layers": config.get("mlp_only_layers", []),
        "rope_theta": config.get("rope_theta", 1000000.0),
        "quantization": config.get("quantization", {"bits": 4, "group_size": 64}),
        "streaming": {
            "pinned_file": "pinned.safetensors",
            "expert_dir": "bin",
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
        }
    }
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(stream_config, f, indent=2)

    # Copy tokenizer files
    import shutil
    for tf in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", 
               "added_tokens.json", "vocab.json", "merges.txt"]:
        src = f"{MLX_MODEL_DIR}/{tf}"
        if os.path.exists(src):
            shutil.copy(src, f"{OUTPUT_DIR}/{tf}")

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s!")
    print(f"Pinned: {pinned_bytes/1e9:.2f} GB")
    print(f"Experts: {total_expert_bytes/1e9:.2f} GB")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
