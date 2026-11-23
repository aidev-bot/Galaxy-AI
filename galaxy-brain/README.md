# Galaxy Brain: otak AI Galaxy, dibuat dari nol

Galaxy Brain adalah model bahasa (Transformer) yang ditulis dan dilatih sendiri.
Ia tidak memakai bobot, API, atau jawaban dari AI lain. Semua yang Galaxy tahu
berasal dari data yang kamu latihkan.

## Jujur dulu: apa yang bisa dan belum bisa

| | Galaxy (budget Rp3 juta) | Claude / ChatGPT Codex |
|---|---|---|
| Ukuran model | 1 – 113 juta parameter | ratusan miliar+ parameter |
| Data latihan | puluhan – ratusan juta token | puluhan triliun token |
| Biaya | ± Rp3 juta + GPU gratis | ratusan juta – miliaran dolar |
| Kemampuan coding | fungsi Python sangat sederhana | program lengkap |

Galaxy **tidak akan setara** Claude atau Codex dengan budget ini. Itu bukan
masalah kode, tapi skala: jumlah komputasi dan data. Yang kamu dapat adalah AI
asli buatanmu sendiri, dengan cara kerja yang sama seperti AI besar, yang bisa
kamu perbesar kalau nanti budget bertambah.

## Hardware dengan budget maksimal Rp3 juta

Harga di bawah adalah perkiraan pasar bekas. Cek ulang di Tokopedia, Shopee,
atau OLX sebelum membeli.

**Pilihan A: sudah punya PC desktop** (paling disarankan)
- Beli GPU bekas **NVIDIA RTX 3060 12GB**, sekitar Rp2,8 – 3,3 juta.
- Syarat PC: slot PCIe x16, PSU minimal 550W dengan kabel 8-pin, RAM 16GB.
- Hasil: bisa melatih preset `kecil` dan `sedang` di rumah.

**Pilihan B: belum punya komputer**
- Beli PC kantor bekas (Core i5 generasi 8–10, RAM 16GB, SSD 512GB), sekitar
  Rp2,5 – 3 juta. Contoh: Dell OptiPlex, HP ProDesk, Lenovo ThinkCentre.
- PC ini untuk menjalankan Galaxy (ngobrol, server).
- Latihan berat dikerjakan di **GPU gratis**:
  - Kaggle Notebooks: GPU T4 gratis, sekitar 30 jam per minggu.
  - Google Colab: GPU T4 gratis, waktunya terbatas.

Dengan Pilihan B pun, latihan bisa jalan tanpa biaya tambahan.

## Cara pakai

Butuh Python 3.10+ dan PyTorch.

```bash
cd galaxy-brain
pip install -r requirements.txt

# 1. Siapkan data (bawaan: kode pustaka standar Python di komputermu)
python siapkan_data.py
#    tambah folder & percakapanmu sendiri:
python siapkan_data.py --sumber ~/proyek-saya --percakapan percakapan.jsonl

# 2. Latih dari nol
python latih.py --ukuran kecil --langkah 20000          # GPU
python latih.py --ukuran mini --langkah 2000            # CPU, untuk mencoba saja

# 3. Galaxy memperbaiki dirinya sendiri (jalankan di Kaggle/Colab/Docker!)
python kembangkan_diri.py --putaran 10

# 4. Pakai
python ngobrol.py              # percakapan
python ngobrol.py --kode       # melanjutkan kode
python server.py               # API di http://127.0.0.1:8765
```

Format `percakapan.jsonl`, satu baris per contoh:

```json
{"user": "Halo, siapa kamu?", "galaxy": "Saya Galaxy, AI buatan sendiri."}
```

### Ukuran model

| Preset | Parameter | Cocok untuk |
|---|---|---|
| `mini` | 0,9 juta | tes di CPU |
| `kecil` | 10,7 juta | RTX 3060 / Kaggle T4, beberapa jam |
| `sedang` | 38,7 juta | RTX 3060 / Kaggle T4, beberapa hari |
| `besar` | 113,5 juta | butuh GPU lebih kuat atau waktu sangat lama |

Kalau memori GPU habis (`CUDA out of memory`), kecilkan `--batch` dan naikkan
`--akumulasi` (contoh: `--batch 8 --akumulasi 2`).

## Bagaimana Galaxy memperbaiki dirinya sendiri

`kembangkan_diri.py` menjalankan siklus ini setiap putaran:

1. Galaxy menulis beberapa jawaban untuk setiap soal di `tugas/latih.jsonl`.
2. Setiap jawaban dijalankan dan diuji otomatis.
3. Jawaban yang lulus disimpan ke ingatan (`models/solusi_benar.jsonl`).
4. Salinan Galaxy dilatih dari jawaban benarnya sendiri, dicampur data lama
   supaya tidak lupa.
5. Salinan diuji dengan soal yang **tidak pernah** dilatihkan (`tugas/uji.jsonl`).
   Salinan menggantikan Galaxy aktif **hanya kalau terbukti lebih baik**.
   Kalau tidak, salinan dibuang.

Jadi Galaxy belajar dari dirinya sendiri, tetapi tidak bisa merusak dirinya.
Riwayat setiap putaran ada di `models/log_kembang.jsonl` dan `models/riwayat/`.

Batasannya:
- Siklus ini hanya berguna kalau Galaxy sudah bisa memecahkan **minimal satu**
  soal. Model yang baru dilatih sebentar belum bisa. Latih lebih lama dulu.
- Galaxy hanya membaik pada jenis soal yang ada di `tugas/`. Tambahkan soal
  sendiri (format JSONL: `id`, `prompt`, `tests`) agar kemampuannya meluas.
  Jangan pernah menyalin soal `uji` ke `latih`, karena nilai ujinya jadi palsu.

**Peringatan keamanan:** langkah 2 menjalankan kode yang ditulis AI.
`galaxy/sandbox.py` membatasi waktu, memori, dan folder, tetapi itu bukan
pengaman penuh. Jalankan `kembangkan_diri.py` di Kaggle, Colab, Docker, atau
VM, jangan di komputer yang berisi data penting.

## Isi folder

```
galaxy/model.py      otak: tokenizer byte + Transformer (RoPE, SwiGLU, RMSNorm)
galaxy/trainer.py    data & pelatihan (AdamW, cosine LR, mixed precision)
galaxy/sandbox.py    menjalankan & menguji kode buatan Galaxy
siapkan_data.py      langkah 1: kumpulkan data
latih.py             langkah 2: latih dari nol / lanjutkan
kembangkan_diri.py   langkah 3: siklus memperbaiki diri sendiri
ngobrol.py           ngobrol di terminal
server.py            API lokal untuk aplikasi lain
tugas/               soal pemrograman + tes otomatis
```
