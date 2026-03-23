#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  HIVEMIND — Local LLM Dashboard                         ║
║  Real-time monitoring & interactive chat for llama.cpp   ║
╚══════════════════════════════════════════════════════════╝
"""

import json
import time
import threading
import sys
import os
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.progress import BarColumn, Progress, TextColumn, SpinnerColumn
from rich.style import Style
from rich.prompt import Prompt

# ─── Config ───────────────────────────────────────────────
SERVER_URL = os.environ.get("LLAMA_URL", "http://localhost:8000")
POLL_INTERVAL = 0.5

# ─── State ────────────────────────────────────────────────
class DashState:
    def __init__(self):
        self.server_status = "connecting..."
        self.model_name = "—"
        self.slots = []
        self.tok_history = deque(maxlen=60)  # last 60 samples (~30s)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.uptime_start = time.time()
        self.last_response = None
        self.last_speed = 0.0
        self.peak_speed = 0.0
        self.chat_log = deque(maxlen=20)
        self.is_generating = False
        self.error = None
        self.last_probe_time = 0
        self.probe_interval = 10  # seconds between heartbeat probes
        self.prompt_speed = 0.0

state = DashState()
console = Console()

# ─── Server Polling ───────────────────────────────────────
def fetch_json(path):
    try:
        req = urllib.request.Request(f"{SERVER_URL}{path}")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def probe_speed():
    """Fire a tiny request to measure current tok/s from the server timings."""
    try:
        payload = json.dumps({
            "model": "qwen3.5-9b",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 20,
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        timings = data.get("timings", {})
        usage = data.get("usage", {})
        speed = timings.get("predicted_per_second", 0)
        prompt_tps = timings.get("prompt_per_second", 0)
        comp_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        if speed > 0:
            state.last_speed = speed
            state.peak_speed = max(state.peak_speed, speed)
            state.tok_history.append(speed)
        if prompt_tps > 0:
            state.prompt_speed = prompt_tps
        state.total_prompt_tokens += prompt_tokens
        state.total_completion_tokens += comp_tokens
        state.total_requests += 1

    except Exception:
        pass

def poll_server():
    last_decoded = {}  # slot_id -> (timestamp, n_decoded)

    while True:
        try:
            health = fetch_json("/health")
            if health and health.get("status") == "ok":
                state.server_status = "online"
                state.error = None
            else:
                state.server_status = "degraded"

            slots = fetch_json("/slots")
            if slots:
                state.slots = slots
                active = sum(1 for s in slots if s.get("is_processing"))
                state.is_generating = active > 0

                # Estimate tok/s by watching decoded token count changes
                now = time.time()
                slot_speeds = []

                for s in slots:
                    sid = s.get("id", 0)
                    processing = s.get("is_processing", False)
                    next_tok = s.get("next_token", [{}])
                    nt = next_tok[0] if (next_tok and isinstance(next_tok, list)) else {}
                    n_decoded = nt.get("n_decoded", 0)

                    if processing and n_decoded > 0:
                        if sid in last_decoded:
                            prev_time, prev_decoded = last_decoded[sid]
                            dt = now - prev_time
                            dn = n_decoded - prev_decoded
                            if dt > 0.05 and dn > 0:
                                slot_tps = dn / dt
                                # Clamp to reasonable range to filter poll jitter
                                if 1.0 < slot_tps < 100.0:
                                    slot_speeds.append(slot_tps)
                        last_decoded[sid] = (now, n_decoded)
                    else:
                        # Slot finished or idle — clear tracking
                        if sid in last_decoded:
                            del last_decoded[sid]

                if slot_speeds:
                    # Use max across slots (they share bandwidth)
                    live_tps = max(slot_speeds)
                    state.last_speed = live_tps
                    state.peak_speed = max(state.peak_speed, live_tps)
                    state.tok_history.append(live_tps)

            # Periodic heartbeat probe when idle to keep stats fresh
            now = time.time()
            if now - state.last_probe_time > state.probe_interval and not state.is_generating:
                state.last_probe_time = now
                probe_speed()

        except Exception as e:
            state.server_status = "offline"
            state.error = str(e)

        # Poll faster when generating, slower when idle
        time.sleep(0.3 if state.is_generating else 1.0)

# ─── Sparkline Generator ─────────────────────────────────
SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values, width=30):
    if not values:
        return "—"
    recent = list(values)[-width:]
    if not recent:
        return "—"
    mn, mx = min(recent), max(recent)
    rng = mx - mn if mx != mn else 1
    return "".join(SPARK_CHARS[min(int((v - mn) / rng * 7), 7)] for v in recent)

# ─── Build Dashboard Layout ──────────────────────────────
def make_header():
    now = datetime.now().strftime("%H:%M:%S")
    uptime = int(time.time() - state.uptime_start)
    h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60

    status_color = {
        "online": "bright_green",
        "degraded": "yellow",
        "offline": "bright_red",
        "connecting...": "dim",
    }.get(state.server_status, "dim")

    bee = "🐝" if state.is_generating else "💤"

    title = Text()
    title.append("  H I V E M I N D  ", style="bold bright_yellow on grey23")
    title.append("  ", style="default")
    title.append(f" {bee} ", style="default")
    title.append(f" {state.server_status.upper()} ", style=f"bold {status_color}")
    title.append(f"  {now}  ", style="dim")
    title.append(f"  uptime {h:02d}:{m:02d}:{s:02d}", style="dim cyan")

    return Panel(
        Align.center(title),
        style="bright_yellow",
        height=3,
    )

def make_model_panel():
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=16)
    table.add_column()

    # Auto-detect model from server slots
    model_name = "—"
    model_params = "—"
    model_arch = "—"
    model_quant = "—"
    model_ctx = "—"
    model_layers = "—"

    if state.slots:
        # Detect from slot data; fallback to server probe
        pass

    # Try to detect from the running server
    props = fetch_json("/props")
    if props:
        mname = props.get("model_alias", "") or props.get("model_path", "")
        if "35B-A3B" in mname:
            model_name = "Qwen3.5-35B-A3B"
            model_params = "34.7B (3B active/tok)"
            model_arch = "MoE + GDN + Attn"
            model_quant = "IQ2_M (10.6 GB)"
            model_ctx = "2,048 / 262K native"
            model_layers = "41/41 (Metal)"
        elif "9B" in mname:
            model_name = "Qwen3.5-9B"
            model_params = "8.95B"
            model_arch = "Hybrid (GDN + Attn)"
            model_quant = "Q4_K_M (5.28 GB)"
            model_ctx = "4,096 / 262K native"
            model_layers = "33/33 (Metal)"
        else:
            model_name = mname.split("/")[-1][:30] if mname else "—"
            model_params = "—"
            model_arch = "—"
            model_quant = "—"
            model_ctx = "—"
            model_layers = "—"

    table.add_row("Model", model_name)
    table.add_row("Parameters", model_params)
    table.add_row("Architecture", model_arch)
    table.add_row("Quantization", model_quant)
    table.add_row("Context", model_ctx)
    table.add_row("GPU Layers", model_layers)
    table.add_row("Backend", "Apple M4 (Metal)")

    return Panel(table, title="[bold cyan]Model Info", border_style="cyan")

def make_perf_panel():
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold green", width=16)
    table.add_column()

    speed = state.last_speed
    peak = state.peak_speed
    spark = sparkline(state.tok_history)

    speed_style = "bold bright_green" if speed > 10 else "bold yellow" if speed > 5 else "bold red"
    table.add_row("Gen tok/s", Text(f"{speed:.1f}", style=speed_style))
    table.add_row("Prompt tok/s", f"{state.prompt_speed:.1f}")
    table.add_row("Peak tok/s", f"{peak:.1f}")
    table.add_row("Throughput", spark)
    table.add_row("", "")
    table.add_row("Requests", str(state.total_requests))
    table.add_row("Prompt Tokens", f"{state.total_prompt_tokens:,}")
    table.add_row("Gen Tokens", f"{state.total_completion_tokens:,}")

    perf_title = "[bold green]Performance [bright_green]● LIVE[/]" if state.is_generating else "[bold green]Performance"
    return Panel(table, title=perf_title, border_style="green")

def make_slots_panel():
    table = Table(box=None, padding=(0, 1))
    table.add_column("Slot", style="bold", width=5)
    table.add_column("Status", width=12)
    table.add_column("Decoded", width=10)
    table.add_column("Remaining", width=10)

    for slot in state.slots[:4]:
        sid = str(slot.get("id", "?"))
        processing = slot.get("is_processing", False)
        next_tok = slot.get("next_token", [{}])
        if next_tok and isinstance(next_tok, list):
            nt = next_tok[0] if next_tok else {}
        else:
            nt = {}
        decoded = nt.get("n_decoded", 0)
        remain = nt.get("n_remain", 0)

        if processing:
            status = Text("● ACTIVE", style="bold bright_green")
            bar_pct = decoded / max(decoded + remain, 1)
            bar_width = 8
            filled = int(bar_pct * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            decoded_str = f"{decoded} {bar}"
        else:
            status = Text("○ idle", style="dim")
            decoded_str = "—"
            remain = "—"

        table.add_row(sid, status, str(decoded_str), str(remain))

    if not state.slots:
        table.add_row("—", Text("waiting...", style="dim"), "—", "—")

    return Panel(table, title="[bold magenta]Inference Slots", border_style="magenta")

def make_chat_panel():
    lines = []
    for entry in state.chat_log:
        role = entry.get("role", "?")
        content = entry.get("content", "")
        speed = entry.get("speed", 0)
        tokens = entry.get("tokens", 0)

        if role == "user":
            lines.append(Text(f"  You: {content}", style="bold bright_white"))
        elif role == "assistant":
            # Truncate long responses
            display = content[:300] + "..." if len(content) > 300 else content
            lines.append(Text(f"  AI: {display}", style="bright_green"))
            if speed > 0:
                lines.append(Text(f"      [{tokens} tokens, {speed:.1f} tok/s]", style="dim green"))
        lines.append(Text(""))

    if not lines:
        lines = [Text("  Send a message to start chatting!", style="dim italic")]

    content = Text("\n").join(lines) if lines else Text("—")

    return Panel(
        content,
        title="[bold yellow]Chat Log",
        border_style="yellow",
        height=14,
    )

def make_help_bar():
    help_text = Text()
    help_text.append("  [c]", style="bold bright_cyan")
    help_text.append(" Chat  ", style="dim")
    help_text.append("[b]", style="bold bright_cyan")
    help_text.append(" Benchmark  ", style="dim")
    help_text.append("[r]", style="bold bright_cyan")
    help_text.append(" Reset Stats  ", style="dim")
    help_text.append("[q]", style="bold bright_cyan")
    help_text.append(" Quit", style="dim")
    return Panel(help_text, style="dim", height=3)

def build_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(make_header(), name="header", size=3),
        Layout(name="body"),
        Layout(make_help_bar(), name="footer", size=3),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    layout["left"].split_column(
        Layout(make_model_panel(), name="model", size=11),
        Layout(make_chat_panel(), name="chat"),
    )

    layout["right"].split_column(
        Layout(make_perf_panel(), name="perf", size=11),
        Layout(make_slots_panel(), name="slots"),
    )

    return layout

# ─── Chat Function ────────────────────────────────────────
def send_chat(message):
    state.chat_log.append({"role": "user", "content": message})
    state.is_generating = True

    try:
        payload = json.dumps({
            "model": "qwen3.5-9b",
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 1000,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        elapsed = time.time() - start
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        comp_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        speed = timings.get("predicted_per_second", comp_tokens / max(elapsed, 0.1))

        state.total_requests += 1
        state.total_prompt_tokens += prompt_tokens
        state.total_completion_tokens += comp_tokens
        state.last_speed = speed
        state.peak_speed = max(state.peak_speed, speed)
        state.tok_history.append(speed)

        state.chat_log.append({
            "role": "assistant",
            "content": content,
            "speed": speed,
            "tokens": comp_tokens,
        })

    except Exception as e:
        state.chat_log.append({
            "role": "assistant",
            "content": f"[Error: {e}]",
            "speed": 0,
            "tokens": 0,
        })
    finally:
        state.is_generating = False

# ─── Benchmark Function ──────────────────────────────────
def run_quick_benchmark():
    prompts = [
        ("Math", "What is 137 * 29? Just give the number."),
        ("Reasoning", "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball? Answer only."),
        ("Code", "Write a Python one-liner to check if a number is prime."),
        ("Knowledge", "What is the speed of light in m/s? Just the number."),
    ]

    state.chat_log.append({"role": "user", "content": "--- BENCHMARK START ---"})

    speeds = []
    for name, prompt in prompts:
        state.chat_log.append({"role": "user", "content": f"[{name}] {prompt}"})
        state.is_generating = True

        try:
            payload = json.dumps({
                "model": "qwen3.5-9b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.3,
            }).encode()

            req = urllib.request.Request(
                f"{SERVER_URL}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            comp_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            speed = timings.get("predicted_per_second", 0)

            state.total_requests += 1
            state.total_prompt_tokens += prompt_tokens
            state.total_completion_tokens += comp_tokens
            state.last_speed = speed
            state.peak_speed = max(state.peak_speed, speed)
            state.tok_history.append(speed)
            speeds.append(speed)

            state.chat_log.append({
                "role": "assistant",
                "content": content[:150],
                "speed": speed,
                "tokens": comp_tokens,
            })

        except Exception as e:
            state.chat_log.append({
                "role": "assistant",
                "content": f"[Error: {e}]",
                "speed": 0,
                "tokens": 0,
            })

    avg = sum(speeds) / len(speeds) if speeds else 0
    state.chat_log.append({
        "role": "user",
        "content": f"--- BENCHMARK DONE: avg {avg:.1f} tok/s ---",
    })
    state.is_generating = False

# ─── Main ─────────────────────────────────────────────────
def main():
    # Start background poller
    poller = threading.Thread(target=poll_server, daemon=True)
    poller.start()

    console.clear()
    console.print(
        Panel(
            Align.center(
                Text.from_markup(
                    "[bold bright_yellow]  H I V E M I N D  [/]\n"
                    "[dim]Local LLM Dashboard[/]\n\n"
                    "[bright_green]Connecting to server...[/]"
                ),
            ),
            border_style="bright_yellow",
            height=8,
        )
    )
    time.sleep(1)

    try:
        with Live(build_dashboard(), refresh_per_second=2, console=console, screen=True) as live:
            while True:
                live.update(build_dashboard())
                time.sleep(0.5)

                # Check for keyboard input (non-blocking)
                # We use a simple approach: check if stdin has data
                import select
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    key = sys.stdin.read(1)

                    if key == "q":
                        break

                    elif key == "c":
                        live.stop()
                        console.print()
                        msg = Prompt.ask("[bold cyan]You")
                        if msg.strip():
                            chat_thread = threading.Thread(
                                target=send_chat, args=(msg,), daemon=True
                            )
                            chat_thread.start()
                        live.start()

                    elif key == "b":
                        bench_thread = threading.Thread(
                            target=run_quick_benchmark, daemon=True
                        )
                        bench_thread.start()

                    elif key == "r":
                        state.total_requests = 0
                        state.total_prompt_tokens = 0
                        state.total_completion_tokens = 0
                        state.peak_speed = 0
                        state.tok_history.clear()
                        state.chat_log.clear()

    except KeyboardInterrupt:
        pass

    console.clear()
    console.print("[bold bright_yellow]HIVEMIND[/] dashboard closed. Server still running.")

if __name__ == "__main__":
    import tty
    import termios

    # Set terminal to raw mode for single-char input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        main()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
