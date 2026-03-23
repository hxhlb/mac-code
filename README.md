# pico-mini

**Run a 35-billion parameter AI agent on a $600 Mac mini — for free.**

A self-contained AI agent with web search, file operations, and code execution, powered entirely by a local LLM running on Apple Silicon. No cloud. No API keys. No subscription.

## What is this?

pico-mini runs **Qwen3.5-35B-A3B** (a 35B-parameter Mixture-of-Experts model with only 3B active per token) locally on a Mac mini M4 via SSD flash-paging — and hits **30 tokens/second**.

| | Speed | Cost | Paging? |
|---|---|---|---|
| **pico-mini (this project)** | **29.8 tok/s** | **$0/hr** | **SSD** |
| NVIDIA + NVMe (Vast.ai) | 1.6 tok/s | $0.44/hr | NVMe |
| NVIDIA + FUSE (RunPod) | 0.075 tok/s | $0.44/hr | Network |
| NVIDIA in-VRAM (RunPod) | 42.5 tok/s | $0.34/hr | No |

Apple Silicon with SSD paging is **18.6x faster** than NVIDIA with NVMe paging.

## Quick Start

### Prerequisites

- Mac with Apple Silicon (M1+, 16GB+ RAM)
- Python 3.10+, Go 1.25+, Homebrew

### 1. Install llama.cpp + download model

```bash
brew install llama.cpp
pip3 install huggingface-hub --break-system-packages

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

### 2. Start the server

```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4
```

### 3. Build the agent backend

```bash
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build
```

### 4. Configure

```bash
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json
```

### 5. Run

```bash
# Agent mode (web search, tools, file ops)
python3 agent.py

# Streaming chat (direct to LLM, no tools)
python3 chat.py

# Dashboard (real-time monitoring)
python3 dashboard.py
```

## Components

| File | What it does |
|---|---|
| `agent.py` | Claude Code-style agent TUI with animated loading, markdown rendering, tool use via PicoClaw |
| `chat.py` | Lightweight streaming chat directly to llama-server |
| `dashboard.py` | Real-time server monitoring — tok/s, slots, memory, sparklines |
| `config.example.json` | PicoClaw config pointing at local llama-server |
| `setup.sh` | One-command install script |

## Agent Features

- **Two modes**: `/agent` (tools + web search) and `/raw` (direct streaming)
- **Animated loading** with live log display while the agent works
- **Markdown rendering** for formatted responses
- **Web search** via DuckDuckGo (no API key needed)
- **URL fetching** — read any webpage
- **Shell execution** — run commands
- **File operations** — read, write, edit, list
- **Session stats** — tok/s, token count, turn tracking

## Architecture

```
┌──────────────────────────────────────┐
│  pico-mini Agent (Python + Rich)     │
├──────────────────────────────────────┤
│  PicoClaw (Go agent framework)       │
│  Tools: search · fetch · exec · MCP  │
├──────────────────────────────────────┤
│  llama.cpp (Metal GPU acceleration)  │
│  OpenAI-compatible API @ :8000       │
├──────────────────────────────────────┤
│  Qwen3.5-35B-A3B (MoE, IQ2_M)       │
│  34.7B params, 3B active per token   │
├──────────────────────────────────────┤
│  Apple M4 · Unified Memory · SSD     │
└──────────────────────────────────────┘
```

## Smaller Hardware?

If you have less RAM or want faster inference, use the dense 9B model instead:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 4096 \
    --n-gpu-layers 99 --reasoning off -t 4
```

The agent auto-detects whichever model is running.

## License

MIT

## Credits

Built with [Qwen3.5](https://huggingface.co/Qwen) (Alibaba), [llama.cpp](https://github.com/ggergov/llama.cpp), [PicoClaw](https://github.com/sipeed/picoclaw), and [Rich](https://github.com/Textualize/rich).
