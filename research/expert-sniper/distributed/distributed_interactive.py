#!/usr/bin/env python3
"""
Interactive Distributed Expert Sniper — multi-turn chat with Qwen3.5-35B.

3-node architecture:
  - Coordinator (this node): pinned model (attention/embeddings/norms) + routing
  - Expert Node A: experts 0-127 in RAM
  - Expert Node B: experts 128-255 in RAM

Usage:
  python distributed_interactive.py \
    --nodes http://<NODE_A_IP>:8301,http://<NODE_B_IP>:8301
"""

import sys
import re
import os
import time
import argparse
import json

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distributed_reader_fast import FastDistributedReader


SNIPER_DIR = os.path.expanduser("~/expert-sniper-mlx")
sys.path.insert(0, SNIPER_DIR)
from moe_agent_35b import run_expert_ffn, MoESniperEngine35B


class InteractiveDistributedEngine:
    """Distributed MoE engine with multi-turn KV cache support."""

    def __init__(self, node_urls):
        self.node_urls = node_urls
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 40

    def load(self):
        import gc
        from mlx.utils import tree_flatten

        MODEL_DIR = os.path.expanduser("~/models/qwen35-stream")

        print("Loading pinned model (attention, embeddings, norms)...")
        t0 = time.time()

        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]

        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
        args = TextModelArgs(
            model_type=config.get("model_type"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            max_position_embeddings=config.get("max_position_embeddings", 262144),
            head_dim=config.get("head_dim"),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            shared_expert_intermediate_size=config.get("shared_expert_intermediate_size"),
            moe_intermediate_size=config["moe_intermediate_size"],
            linear_num_value_heads=config.get("linear_num_value_heads"),
            linear_num_key_heads=config.get("linear_num_key_heads"),
            linear_key_head_dim=config.get("linear_key_head_dim"),
            linear_value_head_dim=config.get("linear_value_head_dim"),
            linear_conv_kernel_dim=config.get("linear_conv_kernel_dim"),
            full_attention_interval=config.get("full_attention_interval"),
            rope_parameters=config.get("rope_parameters"),
        )

        BITS = 4
        GROUP_SIZE = 64
        self.model = TextModel(args)

        from mlx_lm.models.switch_layers import SwitchLinear
        SSM_PROTECT = {"conv1d"}
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding): return True
            if isinstance(module, SwitchLinear): return True
            if not isinstance(module, nn.Linear): return False
            if any(k in path for k in SSM_PROTECT): return False
            if module.weight.shape[-1] < GROUP_SIZE: return False
            return True
        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS,
                     class_predicate=should_quantize)

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        pinned = mx.load(os.path.join(MODEL_DIR, "pinned.safetensors"))
        stripped = [(k.replace("language_model.", "", 1), v) for k, v in pinned.items()]
        self.model.load_weights(stripped, strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters())
                  if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        elapsed = time.time() - t0
        print(f"  Pinned model loaded: {pinned_gb:.1f} GB in {elapsed:.1f}s")

        # Fast distributed reader with connection pooling
        self.reader = FastDistributedReader(
            node_urls=self.node_urls,
            num_layers=self.num_layers,
            num_experts=256,
        )
        print(f"  Connected to {len(self.node_urls)} expert nodes (fast binary transport)")

        self.cache = self.model.make_cache()

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)
        print(f"  Ready!\n")

    def reset_cache(self):
        """Reset KV cache for new conversation."""
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        """Forward pass with distributed expert computation."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        h = self.model.model.embed_tokens(input_ids)
        fa_mask = create_attention_mask(h, self.cache[self.model.model.fa_idx])
        ssm_mask = create_ssm_mask(h, self.cache[self.model.model.ssm_idx])

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask

            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=self.cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=self.cache[i])
            h = h + attn_out
            mx.eval(h)

            normed = layer.post_attention_layernorm(h)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            active_ids = sorted(set(int(e) for e in np.array(inds).flatten()))

            expert_out = self.reader.compute_distributed(
                layer_idx=i,
                expert_ids=active_ids,
                hidden_state=normed,
                top_k_indices=inds,
                top_k_weights=scores,
            )

            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)
            del expert_out, normed, attn_out
            mx.clear_cache()

        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def generate(self, prompt, max_tokens=200, temperature=0.7, show_thinking=True):
        """Generate response with streaming output."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tokens = self.tokenizer.encode(text)

        generated = []
        input_ids = mx.array([tokens])
        t_start = time.time()
        printed_len = 0

        for step in range(max_tokens):
            logits = self.forward(input_ids)
            mx.eval(logits)

            if temperature <= 0:
                next_token = int(mx.argmax(logits[0, -1]).item())
            else:
                probs = mx.softmax(logits[0, -1] / temperature, axis=-1)
                next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            generated.append(next_token)
            input_ids = mx.array([[next_token]])

            # Decode full output and print incremental text
            full = self.tokenizer.decode(generated, skip_special_tokens=False)

            # Handle thinking visibility
            display = full
            if not show_thinking:
                display = re.sub(r'<think>.*?</think>', '', display, flags=re.DOTALL).strip()
                if '<think>' in full and '</think>' not in full:
                    display = ''

            if len(display) > printed_len:
                new_text = display[printed_len:]
                print(new_text, end='', flush=True)
                printed_len = len(display)

            # Speed indicator every 20 tokens
            if (step + 1) % 20 == 0:
                elapsed = time.time() - t_start
                tps = (step + 1) / elapsed
                sys.stdout.write(f' [{tps:.2f} tok/s]')
                sys.stdout.flush()

            # EOS
            if next_token in [248044, 248046]:
                break

        total = time.time() - t_start
        n = len(generated)
        tps = n / total if total > 0 else 0

        print(f'\n\n--- {n} tokens | {total:.1f}s | {tps:.2f} tok/s ---')
        print(f'    {self.reader.stats()}')
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Distributed Expert Sniper")
    parser.add_argument("--nodes", required=True,
                        help="Comma-separated Expert Node URLs")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Hide thinking process")
    args = parser.parse_args()

    node_urls = [u.strip() for u in args.nodes.split(",")]

    print("=" * 60)
    print("  Distributed Expert Sniper — Interactive Mode")
    print(f"  Model: Qwen3.5-35B-A3B (4-bit)")
    print(f"  Nodes: {len(node_urls)} expert partitions")
    for i, url in enumerate(node_urls):
        print(f"    [{i}] {url}")
    print("=" * 60)
    print()

    engine = InteractiveDistributedEngine(node_urls=node_urls)
    engine.load()

    print("Type your message (or 'quit' to exit, 'reset' to clear context)")
    print("-" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if prompt.lower() == "reset":
            engine.reset_cache()
            engine.reader.reads = 0
            engine.reader.read_time = 0.0
            print("Context cleared.")
            continue

        engine.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            show_thinking=not args.no_thinking,
        )


if __name__ == "__main__":
    main()
