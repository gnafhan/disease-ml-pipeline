# PKT Pipeline -- Klasifikasi Penyakit dari Anamnesa

Folder ini terpisah dari notebook/dataset lama di `../` (satu level di atas)
supaya kerjaan baru nggak nyampur sama file lama -- file lama dibiarkan apa
adanya di sana, nggak disentuh.

Restrukturisasi dari notebook lepas-lepas (`v2_v3.ipynb`, `v5_fixed.py`,
`v6_training.py`, `v7_ensemble.py`) jadi satu pipeline modular dengan
protokol evaluasi yang terkunci dan konsisten.

Rancangan lengkapnya (arsitektur, skema versioning, matriks eksperimen) ada
di dokumen "Rancangan Pipeline -- Redo Penuh untuk TA" yang dibuat bareng
Claude, cek chat/artifact terkait.

## Status implementasi (per sesi terakhir)

- [x] `config/experiment.yaml` -- terkunci
- [x] `src/ingest.py` -- **sudah jalan & tervalidasi terhadap raw_data asli**
- [x] `src/label.py` -- **rekonstruksi mapping ICD->kelas, lihat catatan penting di bawah**
- [x] `src/clean.py` -- **sudah jalan & tervalidasi**
- [x] `src/split.py` -- **sudah jalan** (tanpa dependency scikit-learn, lihat catatan)
- [x] `src/pipeline.py` -- **data-v1/v2/v3 SUDAH ter-generate nyata**, lihat `data/processed/`
- [x] `src/train.py` -- kode lengkap, **BELUM PERNAH DIJALANKAN** (butuh GPU, lihat bagian Training)
- [x] `src/evaluate.py` -- kode lengkap, teruji dgn data sintetis
- [x] `tests/` -- 21 test (label/clean/evaluate: unit test murni; ingest: jalan
      terhadap raw_data asli). `pytest` tidak ada di bridge ini (no internet
      utk install) -- jalankan manual dari terminal Mac:
      `source ../.venv/bin/activate && pip install pytest && pytest tests/ -v`
- [x] `reports/generate_report.py` -- sudah jalan, generate kerangka + tabel

## CATATAN PENTING -- sanity-check mapping ICD sebelum treat hasil sbg final

Skrip ASLI yang dulu bikin `DATASET_GABUNGAN_SPLIT_V4/V5b.xlsx` (versi paling
benar di riwayat project ini) **tidak ketemu di notebook manapun** -- semua
notebook lama (`v2_v3.ipynb`, `v5_fixed.py`, dst) cuma MEMAKAI dataset yang
sudah jadi, bukan proses bikinnya.

Jadi `src/label.py` di sini adalah **rekonstruksi ulang** dari:
1. Tabel resmi `ICD SKDR versi 30 Desember 2024 (1).xlsx`
2. Inventarisasi kode ICD nyata di 25 file RSUD NAS + kolom `ICD 10` RS Akademik
3. Cross-check ke kolom `sindrom` RS Akademik (1857 baris) -- untuk SEMUA kelas
   kecuali "Suspek Campak" (yang memang sudah terbukti 95% salah label di
   riwayat project), jumlah hasil mapping ICD **exact match** dengan hitungan
   sindrom aslinya (COVID 1113, AFP 94, Tetanus 24, HFMD 24, Leptospirosis 13,
   Jaundice 123, dst). Validasi kuat, tapi tetap rekonstruksi, bukan skrip asli.

**Cek `data/processed/<version>/dropped_icd_summary.csv`** -- daftar semua
kode ICD yang dibuang (tidak masuk 12 kelas final) beserta jumlah baris &
alasan. Hasil run terakhir: dari 5908 baris mentah, cuma **79 baris dibuang**
(12 kode unik tanpa alasan terdokumentasi -- semua ternyata kombinasi kode gak
relevan spt HIV+hipertensi, cedera kepala, dll, bukan indikasi bug). Sanity
check ke dr. Wawa / dosen sebelum dipakai final, terutama untuk kelas yang
kodenya beda dari tabel resmi (Diare Akut pakai A09 bukan A02, Sindrom
Jaundice Akut pakai R17+B15-19 bukan A95, AFP pakai kode neuro G5x-G8x bukan
cuma A80 -- alasan tiap perbedaan ada di docstring `src/label.py`).

## Data hasil (real, dari raw_data asli -- lihat `data/processed/*/build_summary.json`)

| Versi | Baris total | Train | Val | Test | Kelas |
|-------|:-----------:|:-----:|:---:|:----:|:-----:|
| v1 (ICD-based, belum dibersihkan) | 5829 | 4079 | 548 | 499 | 12 |
| v2 (+ hapus kontrol/post-ranap & COVID incidental) | 5249 | 3673 | 524 | 468 | 12 |
| v3 (+ flag kelas reliable, ambang 30 sampel) | 5249 | 3673 | 524 | 468 | 12 |

Distribusi kelas per versi ada di `data/processed/<version>/class_distribution.json`.

## Setup

1. **Aktifkan venv di terminal Mac langsung** (bukan lewat bridge Claude --
   venv ini pakai path Homebrew macOS, nggak portable ke environment lain).
   Venv-nya masih di folder lama (`../.venv`), sengaja nggak digandain biar
   nggak install ulang semua package:
   ```
   source ../.venv/bin/activate
   ```
2. Cek dependency yang sudah ada:
   ```
   pip list | grep -iE "torch|transformers|pandas|openpyxl|scikit|pyyaml|pytest"
   ```
   Kalau ada yang belum: `pip install -r requirements.txt`
3. **Jangan pernah commit `raw_data/` atau `data/` ke git** -- isinya data
   pasien. Sudah di-`.gitignore`, tapi tetap double-check sebelum push,
   apalagi kalau remote-nya bakal public.

## Regenerasi data (opsional, sudah pernah dijalankan)

```
python -m src.pipeline --data-version v1
python -m src.pipeline --data-version v2
python -m src.pipeline --data-version v3
```
Output: `data/processed/<version>/{train,val,test}.csv` + `class_distribution.json`
+ `dropped_icd_summary.csv` + `build_summary.json`. Format CSV (bukan parquet)
supaya tetap jalan tanpa `pyarrow` di environment mana pun.

## Training -- BUTUH GPU, belum pernah dijalankan di sesi ini

`src/train.py` sudah lengkap (focal loss, class weights, symptom-flag
injection, dropout-after-load fix dari V5, semuanya di-port dari
`v5_fixed.py`) tapi **belum pernah benar-benar dijalankan** -- bridge ini
(3.8GB RAM, no GPU, no internet) dan Mac langsung (CPU-only) keduanya nggak
cukup buat fine-tune IndoBERT-large dalam waktu wajar.

6 kombinasi wajib (data-v1/v2/v3 x model-base/large) -- jalankan satu-satu di
**Kaggle** (GPU T4, gratis, kuota mingguan -- lihat diskusi sebelumnya soal
kelayakan Kaggle utk project modular git-clone):

```
python -m src.train --data-version v1 --model base
python -m src.train --data-version v1 --model large
python -m src.train --data-version v2 --model base
python -m src.train --data-version v2 --model large
python -m src.train --data-version v3 --model base
python -m src.train --data-version v3 --model large
```

Tiap run otomatis ke-log ke `experiments/runs.jsonl` (val_* dan test_* metrics
terpisah dengan jelas -- lihat docstring `src/evaluate.py` soal kenapa ini
penting, ini yang bikin bingung di V6-Final dulu). Sebelum commit ke GPU
penuh, coba `--smoke-test` dulu (1 epoch, subset kecil) buat pastikan kode
jalan tanpa error.

Setelah semua/sebagian run selesai:
```
python -m reports.generate_report
```
Generate `reports/matriks_perbandingan.md` (6 baris, kolom accuracy/precision/
recall/f1 macro & weighted utk val & test, plus f1_macro_reliable_only) dan
`reports/per_kelas_<run_id>.md` per kombinasi.

## Cara run di Kaggle

1. Push folder ini (`pipeline/`) ke GitHub (bisa private repo).
2. Kaggle Notebook baru -> Settings -> Internet: ON.
3. Cell pertama: `!git clone https://github.com/<user>/<repo>.git && cd repo`
   (pakai token via Kaggle Secrets kalau repo private).
4. Upload `data/processed/` sbg Kaggle Dataset terpisah (jangan lewat git --
   ini data pasien, `.gitignore` sudah exclude `data/` dari git tapi Kaggle
   Dataset private OK, atau `!cp -r /kaggle/input/<dataset>/processed data/`).
5. `!pip install -q -r requirements.txt`
6. `!python -m src.train --data-version v1 --model large`
