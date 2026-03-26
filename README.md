# 🍎 mac code

**35B model on a Mac mini. 30 tok/s. $0/month.**

---

## The Breakthrough

A 35 billion parameter model running at 30 tok/s on a $600 Mac mini M4 with 16GB RAM. The model doesn't fit in memory — macOS pages it from the SSD through Apple Silicon's unified memory architecture. No cloud, no API keys, no monthly cost.

Same hardware also runs a 9B model with 64K context (doubled from 32K via quantized KV cache) and persistent context that saves/loads in 0.0003s.

| | **35B MoE (default)** | **9B (extended context)** |
|---|---|---|
| **Backend** | llama.cpp | llama.cpp or MLX |
| **Speed** | 30 tok/s (SSD paging) | 16-20 tok/s |
| **Context** | 12K | 64K (KV cache quantized) |
| **Size** | 10.6 GB (IQ2_M, 2.6 bpw) | 5.3 GB (Q4_K_M) |
| **Memory** | Pages from SSD | Fits in RAM |
| **Persistent KV** | — | Save/load in 0.0003s, R2 sync |
| **TurboQuant** | — | 4x KV cache compression (0.993 cosine) |
| **Best for** | Reasoning, deep analysis | Long context, tool calling, cross-device |

Both run the same `agent.py`. Same slash commands. Same web search. Same shell tools.

---

## Quick Start

### Option A: llama.cpp + 35B MoE (default — 30 tok/s via SSD paging)

```bash
brew install llama.cpp
pip3 install rich ddgs huggingface-hub --break-system-packages

# Download 35B model (10.6 GB — pages from SSD on 16GB Macs)
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"

# Start server
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 12288 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4

# Run agent
python3 agent.py
```

### Option B: MLX + 9B (64K context, persistent KV cache)

```bash
pip3 install mlx-lm rich ddgs --break-system-packages

# Start MLX engine (downloads 9B model on first run)
python3 mlx/mlx_engine.py

# Run agent
python3 agent.py
```

### Option C: llama.cpp + 9B (if you want 64K context without MLX)

```bash
# Download 9B model
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

# Start server (64K context with quantized KV cache)
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4

# Run agent
python3 agent.py
```

---

## Key Findings

### 35B via SSD Paging — 30 tok/s on $600 Hardware

Qwen3.5-35B-A3B (10.6 GB) doesn't fit in 16 GB RAM. macOS pages it from the SSD through Apple Silicon's unified memory. Result: **30 tok/s** on a Mac mini M4. No cloud, no GPU rental, no monthly cost.

### Tool Calling Works at 2.6 Bits Per Weight

The 35B at IQ2_M (2.6 bpw) was supposed to have broken instruction following. JSON function calls DO break. But our LLM-as-Router (simple text classification) works perfectly — 8/8 correct. You don't need JSON tool calling if the model can classify "search" / "shell" / "chat" as plain text.

### Quantized KV Cache Doubles Context — 9B Gets 64K

Two server flags (`--cache-type-k q4_0 --cache-type-v q4_0`) shrink KV cache from 1024 MB to 288 MB. The 9B goes from 32K to 64K context. The 35B goes from 8K to 12K. Zero quality loss on either model.

### TurboQuant — 4x KV Cache Compression (MLX)

Per-group 4-bit quantization of KV cache states, inspired by [Google's TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/). Results: 26.6 MB → 6.5 MB (4.1x compression) at 0.993 cosine similarity. MSE within 2.7x of the Shannon distortion-rate bound. This makes persistent context practical — save once, resume anywhere.

### Persistent Context — Process Once, Resume Anywhere (MLX)

Save KV cache to SSD in 0.04s. Load in 0.0003s (6,677x faster than reprocessing). Upload to Cloudflare R2 for cross-device resume in 1.5s. Combined with TurboQuant, a full 64K context session is 6.5 MB on disk.

### MLX Is 25% Faster for 9B Generation

Same 9B model, same hardware: MLX 20 tok/s vs llama.cpp 16 tok/s on sustained generation. MLX also enables direct tensor access for custom compression research (kv-lab).

---

## How It Works

The LLM classifies its own intent:

```
"find me videos on my desktop"  → LLM says "shell"  → generates find command → executes
"who do the lakers play next"   → LLM says "search" → rewrites query → DuckDuckGo → answers
"explain quantum computing"     → LLM says "chat"   → streams directly
```

Three paths. No hardcoded rules. Upgrading the model upgrades every capability.

---

## MLX Backend — Persistent Context

The MLX backend adds features llama.cpp can't do:

```bash
# Save context after analyzing a codebase
curl -X POST localhost:8000/v1/context/save \
    -d '{"name":"my-project","prompt":"your codebase here"}'

# Next day — resume instantly (0.0003s vs minutes reprocessing)
curl -X POST localhost:8000/v1/context/load \
    -d '{"name":"my-project"}'

# Different Mac — download from R2 (1.5s)
curl -X POST localhost:8000/v1/context/download \
    -d '{"name":"my-project"}'
```

TurboQuant compresses context storage 4x (26.6 MB → 6.7 MB) with 0.993 cosine similarity.

See `mlx/PROJECT.md` for the full research roadmap.

---

## Commands

Type `/` to see all commands:

| Command | Action |
|---|---|
| `/agent` | Agent mode (default) |
| `/raw` | Direct streaming, no tools |
| `/model 9b` | Switch to 9B (64K ctx) |
| `/model 35b` | Switch to 35B MoE (llama.cpp only) |
| `/search <q>` | Quick web search |
| `/bench` | Speed benchmark |
| `/stats` | Session statistics |
| `/cost` | Cost savings vs cloud |
| `/good` / `/bad` | Grade response (self-improvement logging) |
| `/improve` | View grading stats |
| `/clear` | Reset conversation |
| `/quit` | Exit |

---

## Benchmarks

### Agent Tasks — llama.cpp vs MLX (same prompts, same Mac mini M4)

| Task | llama.cpp | MLX | Winner |
|---|---|---|---|
| Shell command | 7.9s | **7.6s** | MLX |
| Math | 12.4s | **9.8s** | **MLX (21%)** |
| Code gen | 12.3s | **9.7s** | **MLX (21%)** |
| Reasoning | 12.3s | **10.0s** | **MLX (19%)** |
| Web search | **45.7s** | 48.3s | llama.cpp |

### Context Persistence (MLX only)

| Operation | Time |
|---|---|
| Reprocess 141 tokens | 1.01s |
| **SSD load** | **0.0003s (6,677x faster)** |
| R2 download + load | 1.5s |
| TurboQuant compress | 26.6 → 6.7 MB (4x) |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  agent.py — LLM-as-Router                        │
│  search / shell / chat                           │
├──────────┬───────────────────────────────────────┤
│ llama.cpp│  MLX backend                          │
│ backend  │  + KV cache save/load                 │
│          │  + TurboQuant 4-bit compression       │
│          │  + Cloudflare R2 sync                 │
│          │  + Paged inference (GPU→SSD)          │
├──────────┴───────────────────────────────────────┤
│  Apple Silicon — Unified Memory + SSD paging     │
└──────────────────────────────────────────────────┘
```

---

## Files

| File | What |
|---|---|
| `agent.py` | CLI agent — works with either backend |
| `chat.py` | Streaming chat |
| `dashboard.py` | Server monitor |
| `web/` | Retro Mac web UI |
| `mlx/mlx_engine.py` | MLX inference server with context API |
| `mlx/kv_cache.py` | KV cache save/load/compress |
| `mlx/r2_store.py` | Cloudflare R2 integration |
| `mlx/turboquant.py` | 4-bit KV compression |
| `mlx/paged_inference.py` | Process docs beyond context limit |
| `mlx/PROJECT.md` | MLX research roadmap |

---

## Scaling

| Mac | RAM | What you can run |
|---|---|---|
| Any Mac (8GB) | 8 GB | 9B, 4K context |
| **Mac mini M4** | **16 GB** | **9B (64K) + 35B MoE (12K, SSD paging)** |
| Mac mini M4 Pro | 48 GB | 35B at Q4 + speculative decoding |
| Mac Studio Ultra | 192 GB | 397B frontier model |

Same `agent.py` at every level. Just swap the model.

---

## Research

This project builds on:
- **[Apple "LLM in a Flash"](https://machinelearning.apple.com/research/efficient-large-language)** — SSD paging via unified memory
- **[Google TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** — KV cache compression
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's native ML framework

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the models
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** — inference engine
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[Cloudflare R2](https://developers.cloudflare.com/r2/)** — free object storage
- **[Rich](https://github.com/Textualize/rich)** — terminal UI

## License

MIT
