#!/usr/bin/env python3
"""
Fast Distributed Expert Reader — persistent HTTP connections + binary transport.

Optimizations over distributed_reader.py:
  1. requests.Session with connection pooling (no TCP handshake per request)
  2. Binary payload (struct header + raw bytes, no base64/JSON bloat)
  3. Pre-allocated numpy buffers to reduce allocation overhead
"""

import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests


# Binary protocol:
#   Request:  [layer_idx:u16][n_experts:u16][expert_ids:u16*N]
#             [hidden_shape:u16*3][indices_shape:u16*3][weights_shape:u16*3]
#             [hidden_bytes][indices_bytes][weights_bytes]
#
#   Response: [n_computed:u16][shape:u16*3][result_bytes (float16)]

HEADER_FMT = "<HH"  # layer_idx, n_experts


class FastDistributedReader:
    """Drop-in replacement for DistributedExpertReader with persistent connections."""

    def __init__(self, node_urls, num_layers=40, num_experts=256):
        self.node_urls = node_urls
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.executor = ThreadPoolExecutor(max_workers=len(node_urls))

        # Persistent HTTP sessions with connection pooling
        self.sessions = []
        for url in node_urls:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1,
                pool_maxsize=1,
                max_retries=2,
            )
            s.mount("http://", adapter)
            self.sessions.append(s)

        # Stats
        self.reads = 0
        self.read_time = 0.0
        self.lru = None  # compatibility

    def _send_compute_binary(self, session, url, payload_bytes):
        """Send binary compute request via persistent connection."""
        resp = session.post(
            f"{url}/compute_bin",
            data=payload_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=60,
        )
        return resp.content

    def _send_compute_json(self, session, url, payload):
        """Fallback JSON endpoint."""
        import base64, json
        resp = session.post(
            f"{url}/compute",
            json=payload,
            timeout=60,
        )
        return resp.json()

    def compute_distributed(self, layer_idx, expert_ids, hidden_state, top_k_indices, top_k_weights):
        """Send hidden state to all nodes, get weighted expert FFN output."""
        import mlx.core as mx

        t0 = time.time()

        # Convert to numpy once
        hs_np = np.array(hidden_state.astype(mx.float16))
        inds_np = np.array(top_k_indices.astype(mx.int32))
        weights_np = np.array(top_k_weights.astype(mx.float32))

        # Build binary payload
        expert_ids_arr = np.array(expert_ids, dtype=np.uint16)
        hs_shape = np.array(hs_np.shape, dtype=np.uint16)
        inds_shape = np.array(inds_np.shape, dtype=np.uint16)
        wt_shape = np.array(weights_np.shape, dtype=np.uint16)

        header = struct.pack("<HH", layer_idx, len(expert_ids))
        payload = (
            header
            + expert_ids_arr.tobytes()
            + hs_shape.tobytes()
            + inds_shape.tobytes()
            + wt_shape.tobytes()
            + hs_np.tobytes()
            + inds_np.tobytes()
            + weights_np.tobytes()
        )

        # Send to all nodes in parallel
        futures = []
        for i, url in enumerate(self.node_urls):
            futures.append(
                self.executor.submit(
                    self._send_compute_binary, self.sessions[i], url, payload
                )
            )

        # Sum results
        total = np.zeros(hs_np.shape, dtype=np.float16)
        total_experts = 0
        for f in futures:
            raw = f.result()
            # Parse response: [n_computed:u16][ndim:u16][shape:u16*ndim][data]
            n_computed = struct.unpack("<H", raw[:2])[0]
            if n_computed > 0:
                ndim = struct.unpack("<H", raw[2:4])[0]
                shape = struct.unpack(f"<{ndim}H", raw[4:4+ndim*2])
                data_start = 4 + ndim * 2
                partial = np.frombuffer(raw[data_start:], dtype=np.float16).reshape(shape)
                total += partial
                total_experts += n_computed

        elapsed = time.time() - t0
        self.reads += total_experts
        self.read_time += elapsed

        return mx.array(total)

    def get_experts(self, layer_idx, expert_ids):
        return {}

    def prefetch_experts(self, layer_idx, expert_ids):
        pass

    def stats(self):
        avg_ms = (self.read_time / max(self.reads, 1)) * 1000
        return (f"distributed: reads={self.reads}, "
                f"avg={avg_ms:.1f}ms/expert, "
                f"total_time={self.read_time:.1f}s")

    def close(self):
        for s in self.sessions:
            s.close()
        self.executor.shutdown(wait=False)
