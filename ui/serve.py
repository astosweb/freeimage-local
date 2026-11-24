"""Tiny local web UI for freeimage-local. Standard library only.

Started by `fi serve`. Talks to sd-server (stable-diffusion.cpp) and saves every image to outputs/.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
SD_URL = os.environ.get("FI_SD_URL", "http://127.0.0.1:7860")
OUT = Path(os.environ.get("FI_OUT", HERE.parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
GPU = threading.Lock()  # one image at a time; the GPU is the bottleneck anyway
NAME = re.compile(r"^[\w.-]+\.png$")
DEFAULT_STEPS = int(os.environ.get("FI_STEPS", 8))


def generate(body: dict) -> dict:
    prompt = str(body.get("prompt", "")).strip()[:2000]
    if not prompt:
        raise ValueError("Write a prompt first.")
    w, h = int(body.get("width", 1024)), int(body.get("height", 1024))
    if not (256 <= w <= 2048 and 256 <= h <= 2048 and w % 64 == 0 and h % 64 == 0):
        raise ValueError("Width and height must be multiples of 64 between 256 and 2048.")
    steps = max(1, min(int(body.get("steps", DEFAULT_STEPS)), 30))
    seed = body.get("seed")
    seed = int(seed) if str(seed or "").isdigit() else random.randint(0, 2**31 - 1)
    req = json.dumps({"prompt": prompt, "width": w, "height": h, "steps": steps,
                      "cfg_scale": 1.0, "seed": seed}).encode()
    with GPU:
        t0 = time.time()
        r = urllib.request.urlopen(urllib.request.Request(
            f"{SD_URL}/sdapi/v1/txt2img", req, {"content-type": "application/json"}), timeout=900)
        png = base64.b64decode(json.loads(r.read())["images"][0])
        secs = time.time() - t0
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}.png"
    (OUT / name).write_bytes(png)
    (OUT / (name + ".json")).write_text(json.dumps(
        {"prompt": prompt, "width": w, "height": h, "steps": steps, "seed": seed, "seconds": round(secs, 1)}))
    return {"url": f"/outputs/{name}", "name": name, "seed": seed, "seconds": round(secs, 1),
            "width": w, "height": h, "prompt": prompt}


def history(limit: int = 60) -> list[dict]:
    items = []
    for p in sorted(OUT.glob("*.png"), reverse=True)[:limit]:
        meta = {}
        try:
            meta = json.loads((OUT / (p.name + ".json")).read_text())
        except (OSError, ValueError):
            pass
        items.append({"url": f"/outputs/{p.name}", "name": p.name, **meta})
    return items


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, code: int, obj):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            page = (HERE / "index.html").read_text()
            if DEFAULT_STEPS == 6:
                page = page.replace('data-v="8" aria-pressed="true"', 'data-v="8" aria-pressed="false"') \
                           .replace('data-v="6" aria-pressed="false"', 'data-v="6" aria-pressed="true"')
            return self.send(200, page.encode(), "text/html; charset=utf-8")
        if path == "/api/history":
            return self.json(200, history())
        if path.startswith("/outputs/") and NAME.match(path[9:]) and (OUT / path[9:]).is_file():
            extra = {"content-disposition": f'attachment; filename="{path[9:]}"'} if "dl=1" in self.path else None
            return self.send(200, (OUT / path[9:]).read_bytes(), "image/png", extra)
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            return self.send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0)) or 0) or b"{}")
            self.json(200, generate(body))
        except ValueError as e:
            self.json(400, {"error": str(e)})
        except Exception as e:  # engine down, timeout…
            self.json(502, {"error": f"Engine error: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    # Flush so headless / nohup / systemd journals show we're up.
    print(f"freeimage UI ready on http://{a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
