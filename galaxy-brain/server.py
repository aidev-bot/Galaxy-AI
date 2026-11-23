"""Server lokal Galaxy, agar otak Galaxy bisa dipakai aplikasi lain (misalnya galaxy-ai.html).

    python server.py                 # http://127.0.0.1:8765

API:
    GET  /api/brain/status
    POST /api/brain/chat       {"pesan": "halo"}         -> {"balasan": "..."}
    POST /api/brain/lengkapi   {"teks": "def f(x):\\n"}  -> {"lanjutan": "..."}
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from galaxy.model import EOS, chat_tokens, decode, encode, load, pick_device

KUNCI = threading.Lock()
STATE = {}


class Handler(BaseHTTPRequestHandler):
    def _kirim(self, kode, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(kode)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._kirim(204, {})

    def do_GET(self):
        if self.path == "/api/brain/status":
            m = STATE["model"]
            return self._kirim(200, {"nama": "Galaxy", "parameter": m.n_params(),
                                     "konteks": m.config.ctx, "meta": STATE["meta"]})
        self._kirim(404, {"error": "tidak ditemukan"})

    def do_POST(self):
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 100_000)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._kirim(400, {"error": "JSON tidak valid"})
        suhu = float(req.get("suhu", 0.7))
        panjang = max(1, min(int(req.get("panjang", 300)), 1000))
        m = STATE["model"]
        if self.path == "/api/brain/chat":
            with KUNCI:
                out = m.generate(chat_tokens(str(req.get("pesan", ""))), panjang, suhu)[0]
            return self._kirim(200, {"balasan": decode(out)})
        if self.path == "/api/brain/lengkapi":
            with KUNCI:
                out = m.generate(encode(str(req.get("teks", ""))), panjang, suhu,
                                 stop_ids=(EOS,))[0]
            return self._kirim(200, {"lanjutan": decode(out)})
        self._kirim(404, {"error": "tidak ditemukan"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aktif.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    STATE["model"], STATE["meta"] = load(a.model, pick_device())
    print(f"Galaxy aktif di http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
