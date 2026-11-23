"""Langkah 1: kumpulkan data latihan Galaxy.

Sumber bawaan: kode sumber pustaka standar Python yang sudah ada di komputermu
(puluhan MB kode Python berlisensi PSF, gratis). Tambahkan folder lain dengan
--sumber, dan percakapan contoh dengan --percakapan.

Contoh:
    python siapkan_data.py
    python siapkan_data.py --sumber ~/proyek-saya ~/catatan --percakapan percakapan.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sysconfig

from galaxy.model import EOS, chat_tokens, encode
from galaxy.trainer import write_tokens

EKSTENSI = {".py", ".md", ".txt", ".js", ".ts", ".html", ".css", ".c", ".h", ".java",
            ".go", ".rs", ".sh", ".json", ".toml", ".yaml", ".yml", ".rst"}
LEWATI = {".git", "node_modules", "__pycache__", "site-packages", ".venv", "venv",
          "data", "models"}


def kumpulkan(folders, max_kb):
    dilihat, docs = set(), []
    for root_dir in folders:
        root_dir = os.path.expanduser(root_dir)
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in LEWATI]
            for name in files:
                if os.path.splitext(name)[1].lower() not in EKSTENSI:
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getsize(path) > max_kb * 1024:
                        continue
                    with open(path, encoding="utf-8") as f:
                        text = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
                if len(text.strip()) < 50 or h in dilihat:
                    continue
                dilihat.add(h)
                docs.append(encode(text) + [EOS])
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", nargs="*", default=[], help="folder tambahan")
    ap.add_argument("--tanpa-python-std", action="store_true",
                    help="jangan pakai kode pustaka standar Python")
    ap.add_argument("--percakapan", help="file .jsonl berisi {\"user\":..., \"galaxy\":...}")
    ap.add_argument("--ulang-percakapan", type=int, default=20,
                    help="berapa kali percakapan diulang agar lebih diingat")
    ap.add_argument("--max-kb", type=int, default=512)
    ap.add_argument("--keluar", default="data")
    a = ap.parse_args()

    folders = list(a.sumber)
    if not a.tanpa_python_std:
        folders.append(sysconfig.get_paths()["stdlib"])
    print("Membaca:", *folders, sep="\n  ")
    docs = kumpulkan(folders, a.max_kb)

    if a.percakapan:
        chats = []
        with open(a.percakapan, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    chats.append(chat_tokens(c["user"], c["galaxy"]))
        docs += chats * a.ulang_percakapan
        print(f"Percakapan: {len(chats)} (diulang {a.ulang_percakapan}x)")

    random.Random(0).shuffle(docs)
    n_val = max(1, len(docs) // 50)
    val, latih = docs[:n_val], docs[n_val:]
    write_tokens(os.path.join(a.keluar, "latih.bin"), (t for d in latih for t in d))
    write_tokens(os.path.join(a.keluar, "val.bin"), (t for d in val for t in d))
    n_tok = sum(map(len, latih))
    print(f"Selesai: {len(docs)} dokumen, {n_tok:,} token latih, "
          f"{sum(map(len, val)):,} token validasi -> {a.keluar}/")


if __name__ == "__main__":
    main()
