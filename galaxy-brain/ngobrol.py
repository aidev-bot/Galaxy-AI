"""Ngobrol dengan Galaxy di terminal.

    python ngobrol.py               # mode percakapan
    python ngobrol.py --kode        # Galaxy melanjutkan kode yang kamu ketik
"""
import argparse

from galaxy.model import EOS, chat_tokens, decode, encode, load, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aktif.pt")
    ap.add_argument("--kode", action="store_true")
    ap.add_argument("--suhu", type=float, default=0.7)
    ap.add_argument("--panjang", type=int, default=300)
    a = ap.parse_args()
    model, meta = load(a.model, pick_device())
    print(f"Galaxy siap ({model.n_params() / 1e6:.1f} juta parameter). Ketik 'keluar' untuk berhenti.")
    while True:
        try:
            teks = input("\nKamu: ")
        except EOFError:
            break
        if teks.strip().lower() == "keluar":
            break
        if a.kode:
            teks = teks.replace("\\n", "\n")
            out = model.generate(encode(teks), a.panjang, a.suhu, stop_ids=(EOS,))[0]
            print(teks + decode(out))
        else:
            out = model.generate(chat_tokens(teks), a.panjang, a.suhu)[0]
            print("Galaxy:", decode(out))


if __name__ == "__main__":
    main()
