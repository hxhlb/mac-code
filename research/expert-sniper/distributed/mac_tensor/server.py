#!/usr/bin/env python3
"""
mac-tensor ui — Web chat UI for the distributed agent.

Serves a single-page HTML chat interface and a Server-Sent Events
endpoint that streams agent events (steps, tool calls, results, final answer).

Usage:
    mac-tensor ui --model gemma4 --nodes http://mac2:8401,http://mac3:8401
    # Then open http://localhost:8500 in your browser
"""

import json
import os
import sys
import time
import threading
from queue import Queue, Empty


def run_server(model_key, node_urls, host="0.0.0.0", port=8500, allow_write=False):
    """Start the FastAPI server with the agent backend pre-loaded."""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from .agent import AgentBackend, run_agent_turn_stream

    print(f"Loading {model_key} distributed engine...")
    backend = AgentBackend(model_key=model_key, node_urls=node_urls)
    backend.load()
    print(f"Backend ready. Connected to {len(node_urls)} expert nodes.")

    app = FastAPI(title="mac-tensor agent UI")

    # Read the static HTML file shipped alongside this server
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    html_path = os.path.join(static_dir, "chat.html")
    with open(html_path) as f:
        chat_html = f.read()

    # Inject backend info into the HTML so the UI can show it
    chat_html = chat_html.replace(
        "{{MODEL_NAME}}",
        {"gemma4": "Gemma 4-26B-A4B", "qwen35": "Qwen 3.5-35B-A3B"}.get(model_key, model_key),
    ).replace(
        "{{NODE_COUNT}}",
        str(len(node_urls)),
    )

    # Lock so only one chat request runs at a time (single MoE engine)
    lock = threading.Lock()

    @app.get("/")
    async def index():
        return HTMLResponse(chat_html)

    @app.get("/api/info")
    async def info():
        return {
            "model": model_key,
            "nodes": node_urls,
            "allow_write": allow_write,
        }

    @app.post("/api/reset")
    async def reset():
        with lock:
            backend.reset()
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)

        max_iterations = int(body.get("max_iterations", 5))
        max_tokens = int(body.get("max_tokens", 300))

        def event_stream():
            # Acquire lock — single concurrent generation
            with lock:
                try:
                    for event in run_agent_turn_stream(
                        backend, message,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens,
                        allow_write=allow_write,
                    ):
                        yield f"data: {json.dumps(event)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    print()
    print("=" * 60)
    print(f"  mac-tensor UI ready")
    print(f"  Open: http://localhost:{port}")
    print(f"        http://{_local_ip()}:{port}  (LAN access)")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _local_ip():
    """Best-effort detection of the LAN IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main(args):
    if not args.nodes:
        print("Error: --nodes is required")
        sys.exit(1)
    node_urls = [u.strip() for u in args.nodes.split(",")]
    run_server(
        model_key=args.model or "gemma4",
        node_urls=node_urls,
        host=args.host or "0.0.0.0",
        port=args.port or 8500,
        allow_write=args.write,
    )
