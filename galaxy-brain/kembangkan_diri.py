"""Langkah 3: Galaxy memperbaiki dirinya sendiri.

Setiap putaran:
  1. Galaxy mencoba menjawab soal-soal pemrograman di tugas/latih.jsonl
     (beberapa percobaan per soal).
  2. Setiap jawaban dijalankan dan diuji secara otomatis (galaxy/sandbox.py).
  3. Jawaban yang LULUS uji disimpan ke "ingatan" (models/solusi_benar.jsonl).
  4. Salinan Galaxy dilatih ulang dari jawaban benarnya sendiri, dicampur
     data lama agar tidak lupa.
  5. Salinan baru diuji dengan soal yang TIDAK PERNAH dilatihkan
     (tugas/uji.jsonl). Hanya jika hasilnya tidak lebih buruk, salinan itu
     menggantikan Galaxy yang aktif. Jika tidak lebih baik, dibuang.

Jadi Galaxy belajar dari dirinya sendiri, tetapi tidak bisa merusak dirinya.

PERINGATAN: langkah 2 menjalankan kode buatan AI. Jalankan di Docker, VM,
Kaggle, atau Colab, bukan di komputer berisi data penting.

    python kembangkan_diri.py --putaran 10
"""
import argparse
import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from galaxy.model import EOS, decode, encode, load, pick_device, save
from galaxy.sandbox import run_tests
from galaxy.trainer import Mixer, docs_to_tensor, eval_loss, read_tokens, train

STOP = re.compile(r"\n(?=\S)")  # baris baru yang tidak menjorok = fungsi selesai


def baca_tugas(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def potong(teks):
    return STOP.split(teks, maxsplit=1)[0].rstrip()


def coba(model, tugas, n, suhu, panjang):
    """Galaxy menulis n jawaban untuk setiap soal. Kembalikan {id: [kode,...]}."""
    hasil = {}
    for t in tugas:
        outs = model.generate(encode(t["prompt"]), panjang, suhu, n=n, stop_ids=(EOS,))
        hasil[t["id"]] = list(dict.fromkeys(potong(decode(o)) for o in outs))
    return hasil


def nilai(tugas, jawaban, pekerja=4):
    """Uji semua jawaban. Kembalikan {id: [kode yang lulus]}."""
    kerja = [(t, k) for t in tugas for k in jawaban[t["id"]] if k.strip()]
    with ThreadPoolExecutor(pekerja) as ex:
        lulus = list(ex.map(lambda tk: run_tests(tk[0]["prompt"] + tk[1], tk[0]["tests"]), kerja))
    benar = {t["id"]: [] for t in tugas}
    for (t, k), ok in zip(kerja, lulus):
        if ok:
            benar[t["id"]].append(k)
    return benar


def skor(model, tugas, panjang):
    """Berapa soal yang benar dengan jawaban pertama (tanpa acak)."""
    jawab = coba(model, tugas, 1, 0.0, panjang)
    return sum(1 for v in nilai(tugas, jawab).values() if v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aktif.pt")
    ap.add_argument("--data", default="data", help="data lama untuk dicampur agar tidak lupa")
    ap.add_argument("--putaran", type=int, default=5)
    ap.add_argument("--percobaan", type=int, default=16, help="jawaban per soal per putaran")
    ap.add_argument("--suhu", type=float, default=0.8)
    ap.add_argument("--panjang", type=int, default=160)
    ap.add_argument("--langkah", type=int, default=300, help="langkah latihan per putaran")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--porsi-baru", type=float, default=0.3,
                    help="porsi data dari jawaban benar sendiri saat latihan")
    a = ap.parse_args()

    dev = pick_device()
    latih_t = baca_tugas("tugas/latih.jsonl")
    uji_t = baca_tugas("tugas/uji.jsonl")
    lama = read_tokens(os.path.join(a.data, "latih.bin"))
    val = read_tokens(os.path.join(a.data, "val.bin"))
    folder = os.path.dirname(os.path.abspath(a.model))
    ingatan_path = os.path.join(folder, "solusi_benar.jsonl")
    log_path = os.path.join(folder, "log_kembang.jsonl")
    os.makedirs(os.path.join(folder, "riwayat"), exist_ok=True)

    ingatan = {}
    if os.path.exists(ingatan_path):
        for r in baca_tugas(ingatan_path):
            ingatan.setdefault(r["id"], [])
            if r["kode"] not in ingatan[r["id"]]:
                ingatan[r["id"]].append(r["kode"])

    model, meta = load(a.model, dev)
    s_uji, s_latih = skor(model, uji_t, a.panjang), skor(model, latih_t, a.panjang)
    v_loss = eval_loss(model, val, dev)
    print(f"Awal: soal uji {s_uji}/{len(uji_t)} | soal latih {s_latih}/{len(latih_t)} | "
          f"val loss {v_loss:.3f}")

    for r in range(1, a.putaran + 1):
        t0 = time.time()
        print(f"\n=== Putaran {r} ===")
        benar = nilai(latih_t, coba(model, latih_t, a.percobaan, a.suhu, a.panjang))
        baru = 0
        with open(ingatan_path, "a", encoding="utf-8") as f:
            for tid, kode_list in benar.items():
                for k in kode_list:
                    if k not in ingatan.setdefault(tid, []):
                        ingatan[tid].append(k)
                        f.write(json.dumps({"id": tid, "kode": k}, ensure_ascii=False) + "\n")
                        baru += 1
        terpecah = sum(1 for v in ingatan.values() if v)
        print(f"Jawaban benar baru: {baru} | soal yang pernah terpecahkan: "
              f"{terpecah}/{len(latih_t)}")
        if terpecah == 0:
            print("Belum ada satu pun soal yang benar. Galaxy masih terlalu lemah:\n"
                  "latih lebih lama dulu dengan latih.py (lebih banyak langkah / data),\n"
                  "atau tambahkan soal yang lebih mudah ke tugas/latih.jsonl.")
            continue

        prompt_by_id = {t["id"]: t["prompt"] for t in latih_t}
        docs = [encode(prompt_by_id[tid] + k + "\n") + [EOS]
                for tid, ks in ingatan.items() if tid in prompt_by_id for k in ks[-4:]]
        sendiri = docs_to_tensor(docs, model.config.ctx + 2)

        calon = copy.deepcopy(model)
        torch.manual_seed(r)
        train(calon, Mixer([(sendiri, a.porsi_baru), (lama, 1 - a.porsi_baru)],
                           calon.config.ctx, seed=r),
              dev, a.langkah, lr=a.lr, bs=a.batch, log_every=max(1, a.langkah // 3))

        c_uji, c_latih = skor(calon, uji_t, a.panjang), skor(calon, latih_t, a.panjang)
        c_loss = eval_loss(calon, val, dev)
        # Harus benar-benar lebih baik: soal uji naik, atau soal uji sama tetapi
        # soal latih naik tanpa merusak kemampuan umum (val loss).
        lebih_baik = (c_uji > s_uji) or (
            c_uji == s_uji and c_latih > s_latih and c_loss <= v_loss * 1.02)
        print(f"Calon: soal uji {c_uji}/{len(uji_t)} (sebelumnya {s_uji}) | "
              f"soal latih {c_latih}/{len(latih_t)} (sebelumnya {s_latih}) | "
              f"val loss {c_loss:.3f} (sebelumnya {v_loss:.3f})")
        save(calon, os.path.join(folder, "riwayat", f"putaran_{r}.pt"), uji=c_uji, latih=c_latih)
        if lebih_baik:
            model, s_uji, s_latih, v_loss = calon, c_uji, c_latih, c_loss
            save(model, a.model, **{**meta, "uji": s_uji, "latih": s_latih, "val_loss": v_loss})
            print("DITERIMA: Galaxy versi baru sekarang aktif.")
        else:
            print("DITOLAK: versi baru tidak lebih baik, Galaxy lama tetap aktif.")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"putaran": r, "diterima": lebih_baik, "uji": c_uji,
                                "latih": c_latih, "val_loss": round(c_loss, 4),
                                "detik": round(time.time() - t0)}) + "\n")


if __name__ == "__main__":
    main()
