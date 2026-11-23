"""Kirim dan jalankan Galaxy di GPU T4 gratis Kaggle, lalu ambil hasilnya.

Butuh: pip install kaggle, dan API key Kaggle (~/.kaggle/kaggle.json atau
variabel KAGGLE_USERNAME / KAGGLE_KEY). Akun Kaggle harus sudah verifikasi
nomor HP agar boleh memakai GPU.

    python kaggle_jalankan.py kirim --pemilik USERNAME --mode cepat   # tes ±10 menit
    python kaggle_jalankan.py kirim --pemilik USERNAME --mode penuh   # latihan berjam-jam
    python kaggle_jalankan.py status --pemilik USERNAME
    python kaggle_jalankan.py ambil  --pemilik USERNAME               # unduh ke hasil_kaggle/
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

SINI = os.path.dirname(os.path.abspath(__file__))
FILES = ["galaxy/__init__.py", "galaxy/model.py", "galaxy/trainer.py", "galaxy/sandbox.py",
         "siapkan_data.py", "latih.py", "kembangkan_diri.py", "ngobrol.py", "server.py",
         "tugas/latih.jsonl", "tugas/uji.jsonl"]

# Langkah yang dijalankan di Kaggle untuk tiap mode.
MODES = {
    "cepat": [
        "python siapkan_data.py --keluar /tmp/data",
        "python latih.py --ukuran mini --langkah 100000 --batas-menit 3 --eval-tiap 500 "
        "--data /tmp/data --keluar {out}/models/aktif.pt",
        "python kembangkan_diri.py --model {out}/models/aktif.pt --data /tmp/data "
        "--putaran 1 --percobaan 4 --langkah 50",
    ],
    "penuh": [
        "python siapkan_data.py --keluar /tmp/data",
        "python latih.py --ukuran sedang --langkah 1000000 --batas-menit 330 --batch 16 "
        "--eval-tiap 1000 --data /tmp/data --keluar {out}/models/aktif.pt",
        "python kembangkan_diri.py --model {out}/models/aktif.pt --data /tmp/data "
        "--putaran 12 --percobaan 16 --langkah 300",
    ],
}

RUNNER = r'''
import base64, io, os, subprocess, sys, tarfile, time
OUT = os.environ.get("GALAXY_OUT", "/kaggle/working")
SRC = "/tmp/galaxy-brain"
BUNDLE = "{bundle}"
STEPS = {steps!r}
os.makedirs(SRC, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(SRC)
os.makedirs(OUT + "/models", exist_ok=True)
subprocess.run("nvidia-smi; python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'",
               shell=True)
log = open(OUT + "/log.txt", "a")
for cmd in STEPS:
    cmd = cmd.format(out=OUT)
    print("\n$ " + cmd, flush=True); log.write("\n$ " + cmd + "\n"); log.flush()
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=SRC, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line); log.write(line); log.flush()
    p.wait()
    msg = "[selesai dalam %.0f menit, kode %d]" % ((time.time() - t0) / 60, p.returncode)
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if p.returncode != 0:
        sys.exit(p.returncode)
subprocess.run("rm -rf %s/models/riwayat" % OUT, shell=True)  # hemat ukuran unduhan
'''


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            tar.add(os.path.join(SINI, f), arcname=f)
    return base64.b64encode(buf.getvalue()).decode()


def kaggle(*args):
    print("$ kaggle", *args, flush=True)
    return subprocess.run(["kaggle", *args], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aksi", choices=["kirim", "status", "ambil"])
    ap.add_argument("--pemilik", required=True, help="username Kaggle kamu")
    ap.add_argument("--nama", default="galaxy-brain")
    ap.add_argument("--mode", default="cepat", choices=list(MODES))
    ap.add_argument("--keluar", default="hasil_kaggle")
    a = ap.parse_args()
    ref = f"{a.pemilik}/{a.nama}"

    if a.aksi == "status":
        sys.exit(kaggle("kernels", "status", ref).returncode)
    if a.aksi == "ambil":
        os.makedirs(a.keluar, exist_ok=True)
        sys.exit(kaggle("kernels", "output", ref, "-p", a.keluar, "-o").returncode)

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "run.py"), "w") as f:
            f.write(RUNNER.replace("{bundle}", bundle()).replace("{steps!r}", repr(MODES[a.mode])))
        meta = {"id": ref, "title": a.nama, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        sys.exit(kaggle("kernels", "push", "-p", d).returncode)


if __name__ == "__main__":
    main()
