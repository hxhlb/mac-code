# Claude Code Instructions — Distributed Expert Sniper (mac-tensor)

## What This Is

Distributed MoE inference across multiple Macs over the network. Each Mac holds a partition of expert FFN weights in RAM. A coordinator Mac runs attention/routing and dispatches expert compute to the partition nodes via HTTP.

Supports: **Qwen 3.5-35B-A3B** and **Gemma 4-26B-A4B** (both 4-bit quantized).

## Single Machine Alternative

If the user only has one Mac, point them to `../mlx-sniper/` instead — it streams experts from SSD at **5.37 tok/s** on a single M4 Mac Mini (16 GB). Much faster than distributed for single-machine use.

```bash
cd ../mlx-sniper && pip install -e .
mlx-sniper chat ~/models/qwen3-30b
```

## Architecture

```
Coordinator (Mac 1): pinned model (~1.5 GB) — attention, embeddings, norms, router
    ├── Expert Node A (Mac 2): experts 0-N in RAM (~9-13 GB)
    └── Expert Node B (Mac 3): experts N+1-M in RAM (~9-13 GB)
```

## How to Spin Up (Step by Step)

### Prerequisites (all Macs)
```bash
pip install mlx mlx-lm numpy fastapi uvicorn requests
pip install transformers tokenizers huggingface_hub
```

### Qwen 3.5-35B Setup

#### 1. Download model (on each Mac)
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-35B-A3B-4bit', local_dir='~/models/qwen35-4bit')
"
```

#### 2. Split model (on each Mac)
```bash
python3 split_qwen.py
# Edit MODEL_DIR and OUTPUT_DIR paths in the script first
# Creates: ~/models/qwen35-stream/{pinned.safetensors, bin/layer_XX.bin, config.json}
```

#### 3. Start expert nodes
```bash
# Mac 2 (experts 0-127):
python3 expert_node_fast.py --partition 0-127 --model-dir ~/models/qwen35-stream --port 8301

# Mac 3 (experts 128-255):
python3 expert_node_fast.py --partition 128-255 --model-dir ~/models/qwen35-stream --port 8301
```
Wait ~90 seconds for expert loading. Verify with `curl http://<MAC_IP>:8301/health`.

#### 4. Run coordinator (Mac 1)
```bash
python3 distributed_interactive.py \
  --nodes http://<MAC2_IP>:8301,http://<MAC3_IP>:8301 \
  --max-tokens 300
```

### Gemma 4-26B Setup

#### 1. Download model (on each Mac)
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit', local_dir='~/models/gemma4-4bit')
"
```

#### 2. Start expert nodes (no split needed — loads from safetensors directly)
```bash
# Mac 2 (experts 0-63):
python3 gemma4_expert_node.py --partition 0-63 --model-dir ~/models/gemma4-4bit --port 8401

# Mac 3 (experts 64-127):
python3 gemma4_expert_node.py --partition 64-127 --model-dir ~/models/gemma4-4bit --port 8401
```
Wait ~80 seconds. Verify with `curl http://<MAC_IP>:8401/health`.

#### 3. Run coordinator (Mac 1)
The Gemma 4 coordinator requires the custom model file `models_gemma4.py` to be importable.
```bash
python3 gemma4_distributed.py \
  --nodes http://<MAC2_IP>:8401,http://<MAC3_IP>:8401 \
  --max-tokens 200
```

## File Map

| File | Role | Runs On |
|------|------|---------|
| `expert_node_fast.py` | Qwen expert partition server (FastAPI + binary) | Expert Macs |
| `gemma4_expert_node.py` | Gemma 4 expert partition server | Expert Macs |
| `distributed_reader_fast.py` | Client with connection pooling (requests.Session) | Coordinator |
| `distributed_interactive.py` | Qwen interactive chat coordinator | Coordinator |
| `gemma4_distributed.py` | Gemma 4 interactive chat coordinator | Coordinator |
| `split_qwen.py` | Splits Qwen MLX model into pinned + expert bins | Any Mac |
| `split_gemma4.py` | Splits Gemma 4 MLX model into pinned + expert bins | Any Mac (optional) |
| `models_gemma4.py` | Custom Gemma 4 model definition for MLX | Coordinator |

## Key Details

- Expert nodes serve `/compute_bin` (binary) and `/compute` (JSON) endpoints
- Binary transport: struct-packed headers + raw float16/int32/float32 payloads
- Coordinator sends hidden states to ALL nodes in parallel per layer
- Each node computes only its partition's experts via `gather_qmm`
- Results are summed on the coordinator

## Current Performance (Scaleway Mac Mini M2 × 3)

| Model | Speed | Notes |
|-------|-------|-------|
| Qwen 3.5-35B | 1.30 tok/s | 256 experts, 40 layers |
| Gemma 4-26B | 1.23 tok/s | 128 experts, 30 layers |

Bottleneck: 30-40 sequential HTTP round trips per token (~12ms each).

## Troubleshooting

- **Port in use**: `lsof -i :<PORT> -t | xargs kill -9`
- **Process names**: Python shows as `Python` (capital P) on macOS, not `python3`. Use `lsof` or `kill` by PID.
- **Timeout errors**: Increase timeout in `distributed_reader_fast.py` (default 60s)
- **Wrong quantization**: Gemma 4 uses mixed quant — MLP/router at 8-bit, rest at 4-bit. The coordinator handles this automatically.
- **Memory**: Each expert node needs ~10-13 GB. Set `--memory-limit-gb 14` (default).
