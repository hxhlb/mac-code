#!/usr/bin/env python3
"""
pico-mini — Local AI agent on a Mac mini
"""

import json, sys, os, time, subprocess, re, threading, select
import urllib.request

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding

SERVER = os.environ.get("LLAMA_URL", "http://localhost:8000")
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
console = Console()

# ── ANSI strip ─────────────────────────────────────
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(text):
    return ANSI_RE.sub('', text)

# ── pixel creature frames ─────────────────────────
CREATURES = [
    # walking creature
    [
        "   ᕕ( ᐛ )ᕗ  ",
        "  ᕕ( ᐛ )ᕗ   ",
        " ᕕ( ᐛ )ᕗ    ",
        "  ᕕ( ᐛ )ᕗ   ",
        "   ᕕ( ᐛ )ᕗ  ",
        "    ᕕ( ᐛ )ᕗ ",
        "     ᕕ( ᐛ )ᕗ",
        "    ᕕ( ᐛ )ᕗ ",
    ],
    # little spider/crab
    [
        "  /\\_/\\  ",
        "  /\\_/\\  ",
        " ( o.o ) ",
        " ( o.o ) ",
        "  > ^ <  ",
        "  > ^ <  ",
        " ( o.o ) ",
        " ( o.o ) ",
    ],
    # simple dot bounce
    [
        "  ⠋  ",
        "  ⠙  ",
        "  ⠹  ",
        "  ⠸  ",
        "  ⠼  ",
        "  ⠴  ",
        "  ⠦  ",
        "  ⠧  ",
        "  ⠇  ",
        "  ⠏  ",
    ],
]

import random
CREATURE = CREATURES[random.randint(0, len(CREATURES) - 1)]

# ── live working display ──────────────────────────
class WorkingDisplay:
    """Shows animated creature + live log lines while agent works."""

    def __init__(self):
        self.logs = []
        self.frame = 0
        self.start_time = time.time()
        self.done = False
        self.phase = "thinking"

    def add_log(self, line):
        clean = strip_ansi(line).strip()
        if not clean:
            return
        # Parse picoclaw log lines for interesting events
        lower = clean.lower()
        if any(k in lower for k in [
            "processing message", "routed message", "turn_start",
            "llm_request", "tool_call", "tool_result", "web_search",
            "web_fetch", "exec", "turn_end", "context_compress",
            "duckduckgo", "fetch", "read_file", "write_file",
        ]):
            # Extract the interesting part
            if "Processing message" in clean:
                self.phase = "reading your message"
            elif "llm_request" in clean:
                self.phase = "thinking"
            elif "tool_call" in clean or "web_search" in lower:
                self.phase = "searching the web"
            elif "web_fetch" in lower or "fetch" in lower:
                self.phase = "fetching page"
            elif "exec" in lower:
                self.phase = "running command"
            elif "read_file" in lower:
                self.phase = "reading file"
            elif "write_file" in lower:
                self.phase = "writing file"
            elif "context_compress" in lower:
                self.phase = "compressing context"
            elif "turn_end" in lower:
                self.phase = "finishing up"

            # Keep last 3 log lines
            short = clean
            # Trim timestamps and prefixes
            if ">" in short:
                short = short.split(">", 1)[-1].strip()
            if len(short) > 80:
                short = short[:77] + "..."
            self.logs.append(short)
            if len(self.logs) > 3:
                self.logs.pop(0)

    def render(self):
        self.frame += 1
        elapsed = time.time() - self.start_time

        creature_frame = CREATURE[self.frame % len(CREATURE)]

        t = Text()
        t.append(f"  {creature_frame}", style="bright_cyan")
        t.append(f"  {self.phase}", style="bold bright_cyan")
        t.append(f"  {elapsed:.0f}s", style="dim")
        t.append("\n")

        for log in self.logs[-3:]:
            t.append(f"  {log}\n", style="dim italic")

        return t

# ── detect model ───────────────────────────────────
def detect_model():
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "Qwen3.5-35B-A3B", "MoE 34.7B · 3B active · IQ2_M"
        elif "9B" in alias:
            return "Qwen3.5-9B", "8.95B dense · Q4_K_M"
        return alias.replace(".gguf", "").split("/")[-1], "local"
    except Exception:
        return "offline", ""

# ── streaming chat (raw mode) ─────────────────────
def stream_llm(messages):
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    start = time.time()
    tokens = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        while True:
            ch = resp.read(1)
            if not ch:
                break
            buf += ch.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    return full, tokens, time.time() - start
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full += c
                        tokens += 1
                        yield c
                except Exception:
                    pass

    return full, tokens, time.time() - start

# ── picoclaw agent call with live logs ─────────────
BANNER_PATTERNS = ["██", "╔═", "╚═", "╝", "║", "picoclaw"]

def is_banner_line(line):
    clean = strip_ansi(line).strip()
    if not clean:
        return True
    for pat in BANNER_PATTERNS:
        if pat in clean:
            return True
    return False

def picoclaw_call_live(message, session="pico-mini"):
    """Run picoclaw with live animated display showing logs."""
    cmd = [PICOCLAW, "agent", "-m", message, "-s", session]
    display = WorkingDisplay()
    result_holder = [None]  # mutable container for thread result

    def run_picoclaw():
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            result_holder[0] = r
            # Feed stderr to display for log updates
            if r.stderr:
                for line in r.stderr.split("\n"):
                    display.add_log(line)
        except subprocess.TimeoutExpired:
            result_holder[0] = None

    worker = threading.Thread(target=run_picoclaw, daemon=True)
    worker.start()

    # Animate while subprocess runs
    with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
        while worker.is_alive():
            live.update(display.render())
            time.sleep(0.15)

    worker.join(timeout=1)

    result = result_holder[0]
    if result is None:
        return "[Timeout]", display.phase

    # Parse: strip ANSI, find lobster emoji, extract response
    clean = strip_ansi(result.stdout)
    idx = clean.find("\U0001f99e")
    if idx >= 0:
        response = clean[idx:].lstrip("\U0001f99e").strip()
    else:
        # Fallback: take everything after the banner block
        lines = clean.split("\n")
        response_lines = []
        past_banner = False
        for line in lines:
            s = line.strip()
            if not past_banner:
                if not s or any(c in s for c in ["██", "╔", "╚", "╝", "║"]):
                    continue
                past_banner = True
            if past_banner and s:
                response_lines.append(s)
        response = "\n".join(response_lines).strip()

    return response, display.phase

# ── banner ─────────────────────────────────────────
def print_banner(model_name, model_detail):
    console.print()
    logo = Text()
    logo.append("  pico", style="bold bright_cyan")
    logo.append("-", style="dim")
    logo.append("mini", style="bold bright_yellow")
    console.print(logo)

    sub = Text()
    sub.append("  local AI agent on a Mac mini", style="dim italic")
    console.print(sub)
    console.print()

    rows = [
        ("model", model_name, model_detail),
        ("tools", "search · fetch · exec · files", ""),
        ("cost", "$0.00/hr", "Apple M4 Metal · localhost:8000"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:6s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)

    console.print()
    console.print(Rule(style="dim"))
    console.print()

# ── render helpers ─────────────────────────────────
def render_speed(tokens, elapsed):
    if elapsed <= 0 or tokens <= 0:
        return
    speed = tokens / elapsed
    clr = "bright_green" if speed > 20 else "yellow" if speed > 10 else "red"
    s = Text()
    s.append(f"  {speed:.1f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
    console.print(s)

# ── commands ───────────────────────────────────────
def print_help():
    cmds = [
        ("/agent", "Agent mode — tools enabled (default)"),
        ("/raw", "Raw mode — direct streaming, no tools"),
        ("/clear", "Clear conversation"),
        ("/stats", "Session statistics"),
        ("/model", "Current model info"),
        ("/tools", "List available tools"),
        ("/system <msg>", "Set system prompt"),
        ("/quit", "Exit"),
    ]
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold bright_cyan", width=16)
    t.add_column(style="dim")
    for cmd, desc in cmds:
        t.add_row(cmd, desc)
    console.print(t)
    console.print()

# ── main ───────────────────────────────────────────
def main():
    model_name, model_detail = detect_model()
    console.clear()
    print_banner(model_name, model_detail)

    messages = []
    session_tokens = 0
    session_time = 0.0
    session_turns = 0
    session_id = f"pm-{int(time.time())}"
    use_agent = True

    while True:
        try:
            tag = "agent" if use_agent else "raw"
            console.print(f"  [dim]{tag}[/] [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/clear":
            messages.clear()
            session_id = f"pm-{int(time.time())}"
            console.clear()
            print_banner(model_name, model_detail)
            console.print("  [dim]cleared.[/]\n")
            continue
        elif cmd == "/stats":
            avg = session_tokens / session_time if session_time > 0 else 0
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_column(style="bold bright_cyan", width=12)
            t.add_column()
            t.add_row("turns", str(session_turns))
            t.add_row("tokens", f"{session_tokens:,}")
            t.add_row("time", f"{session_time:.1f}s")
            t.add_row("avg speed", f"{avg:.1f} tok/s")
            t.add_row("mode", tag)
            console.print(t)
            console.print()
            continue
        elif cmd == "/model":
            model_name, model_detail = detect_model()
            console.print(f"  [bold white]{model_name}[/]  [dim]{model_detail}[/]\n")
            continue
        elif cmd == "/tools":
            for name, desc in [
                ("web_search", "DuckDuckGo"), ("web_fetch", "read URLs"),
                ("exec", "shell commands"), ("read_file", "local files"),
                ("write_file", "create files"), ("edit_file", "modify files"),
                ("list_dir", "browse dirs"), ("subagent", "spawn tasks"),
            ]:
                t = Text()
                t.append("  ▸ ", style="bright_cyan")
                t.append(name, style="bold bright_cyan")
                t.append(f"  {desc}", style="dim")
                console.print(t)
            console.print()
            continue
        elif cmd == "/agent":
            use_agent = True
            console.print("  [dim]agent mode (tools enabled)[/]\n")
            continue
        elif cmd == "/raw":
            use_agent = False
            console.print("  [dim]raw mode (streaming, no tools)[/]\n")
            continue
        elif cmd in ("/help", "/?"):
            print_help()
            continue
        elif cmd.startswith("/system "):
            sys_msg = user_input[8:].strip()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = sys_msg
            else:
                messages.insert(0, {"role": "system", "content": sys_msg})
            console.print(f"  [dim italic]system: {sys_msg[:80]}[/]\n")
            continue

        console.print()

        # ── agent mode ─────────────────────────────
        if use_agent:
            start = time.time()
            response, final_phase = picoclaw_call_live(user_input, session=session_id)
            elapsed = time.time() - start

            if response:
                console.print()
                # Render as markdown if it contains markdown-like content
                if any(c in response for c in ["##", "**", "```", "- ", "1. ", "* "]):
                    console.print(Padding(Markdown(response), (0, 2)))
                else:
                    for line in response.split("\n"):
                        console.print(f"  {line}")
                console.print()
                tokens_est = len(response.split())
                render_speed(tokens_est, elapsed)
                session_tokens += tokens_est
                session_time += elapsed
                session_turns += 1
            else:
                console.print("  [bold red]no response[/]")

        # ── raw streaming mode ─────────────────────
        else:
            messages.append({"role": "user", "content": user_input})
            full = ""
            tokens = 0
            start = time.time()

            try:
                # Show creature while waiting for first token
                display = WorkingDisplay()
                display.phase = "thinking"
                first_token = True

                with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
                    gen = stream_llm(messages)
                    for chunk in gen:
                        if isinstance(chunk, str):
                            if first_token:
                                first_token = False
                                live.stop()
                                console.print("  ", end="")
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1
                        else:
                            display.frame += 1
                            live.update(display.render())

                elapsed = time.time() - start
                # If the response has markdown, re-render it nicely
                if any(c in full for c in ["##", "**", "```", "- ", "1. "]):
                    # Clear the raw streamed text and re-render as markdown
                    console.print("\n")
                    console.print(Padding(Markdown(full), (0, 2)))
                else:
                    console.print("\n")
                render_speed(tokens, elapsed)
                session_tokens += tokens
                session_time += elapsed
                session_turns += 1
                messages.append({"role": "assistant", "content": full})

            except Exception as e:
                console.print(f"  [bold red]{e}[/]")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

        console.print()

    # ── exit ───────────────────────────────────────
    console.print()
    if session_turns > 0:
        avg = session_tokens / session_time if session_time > 0 else 0
        console.print(
            f"  [bold bright_cyan]pico[/][dim]-[/][bold bright_yellow]mini[/]"
            f"  [dim]{session_turns} turns · {session_tokens:,} tokens · {avg:.1f} tok/s[/]"
        )
    console.print()

if __name__ == "__main__":
    main()
