"""Menjalankan kode buatan Galaxy secara terisolasi untuk mengecek benar/salah.

PERINGATAN: ini isolasi dasar (proses terpisah, batas waktu, batas memori,
folder sementara). Ini BUKAN pengaman penuh. Jalankan kembangkan_diri.py di
dalam Docker, VM, Kaggle, atau Colab, bukan di komputer berisi data penting.
"""
import os
import subprocess
import sys
import tempfile

try:
    import resource  # hanya ada di Linux/macOS
except ImportError:  # Windows
    resource = None


def _limits(timeout):
    def apply():
        mem = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        os.setsid()
    return apply


def run_tests(code: str, tests: str, timeout: int = 5) -> bool:
    """True jika `code` lalu `tests` berjalan tanpa error."""
    with tempfile.TemporaryDirectory(prefix="galaxy_uji_") as d:
        path = os.path.join(d, "solusi.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests + "\n")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-S", path], cwd=d, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout, preexec_fn=_limits(timeout) if resource else None)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
