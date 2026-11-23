"""Langkah 2: latih otak Galaxy dari nol (atau lanjutkan latihan).

Contoh:
    python latih.py --ukuran kecil --langkah 5000          # mulai dari nol
    python latih.py --lanjut models/aktif.pt --langkah 5000 # lanjutkan
"""
import argparse
import os

import torch

from galaxy.model import PRESETS, Config, Galaxy, load, pick_device, save
from galaxy.trainer import Mixer, read_tokens, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukuran", default="kecil", choices=list(PRESETS))
    ap.add_argument("--lanjut", help="checkpoint untuk dilanjutkan")
    ap.add_argument("--data", default="data")
    ap.add_argument("--keluar", default="models/aktif.pt")
    ap.add_argument("--langkah", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--akumulasi", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-tiap", type=int, default=500)
    ap.add_argument("--perangkat", default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile (lebih cepat di GPU)")
    a = ap.parse_args()

    dev = pick_device(a.perangkat)
    torch.manual_seed(0)
    if a.lanjut:
        model, meta = load(a.lanjut, dev)
        mulai = meta.get("langkah_total", 0)
        print(f"Melanjutkan {a.lanjut} (sudah {mulai} langkah)")
    else:
        model, mulai = Galaxy(Config(**PRESETS[a.ukuran])).to(dev), 0
    lr = a.lr or (1e-4 if a.lanjut else 6e-4)
    print(f"Perangkat: {dev} | parameter: {model.n_params() / 1e6:.1f} juta | "
          f"konteks: {model.config.ctx}")

    latih = read_tokens(os.path.join(a.data, "latih.bin"))
    val = read_tokens(os.path.join(a.data, "val.bin"))
    print(f"Data: {len(latih):,} token latih, {len(val):,} token validasi")

    net = torch.compile(model) if a.compile else model
    terbaik = [float("inf")]

    def simpan(step, val_loss):
        if val_loss < terbaik[0]:
            terbaik[0] = val_loss
            save(model, a.keluar, langkah_total=mulai + step, val_loss=val_loss)
            print(f"  disimpan -> {a.keluar}")

    train(net, Mixer([(latih, 1.0)], model.config.ctx), dev, a.langkah, lr=lr, bs=a.batch,
          accum=a.akumulasi, val_data=val, eval_every=a.eval_tiap, save_fn=simpan)
    print(f"Selesai. Val loss terbaik: {terbaik[0]:.3f}")


if __name__ == "__main__":
    main()
