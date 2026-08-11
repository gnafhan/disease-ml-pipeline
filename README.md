# PKT Pipeline -- Klasifikasi Penyakit dari Anamnesa

Restrukturisasi dari notebook lepas-lepas (`v2_v3.ipynb`, `v5_fixed.py`,
`v6_training.py`, `v7_ensemble.py`) jadi satu pipeline modular dengan
protokol evaluasi yang terkunci dan konsisten.

Rancangan lengkapnya (arsitektur, skema versioning, matriks eksperimen) ada
di dokumen "Rancangan Pipeline -- Redo Penuh untuk TA" yang dibuat bareng
Claude, cek chat/artifact terkait.

## Setup

1. **Aktifkan venv di terminal Mac langsung** (bukan lewat bridge Claude --
   venv ini pakai path Homebrew macOS, nggak portable ke environment lain):
   ```
   source .venv/bin/activate
   ```
2. Cek dependency yang sudah ada:
   ```
   pip list | grep -iE "torch|transformers|pandas|openpyxl|scikit"
   ```
   Kemungkinan besar semua udah terinstall (dipakai training sebelumnya).
   Kalau ada yang belum: `pip install -r requirements.txt`
3. **Jangan pernah commit `raw_data/` atau `data/` ke git** -- isinya data
   pasien. Sudah di-`.gitignore`, tapi tetap double-check sebelum push,
   apalagi kalau remote-nya bakal public.

## Urutan kerja

1. `config/experiment.yaml` -- sudah diisi dengan protokol yang dikunci
   (split 70/15/15 seed 42, 12 kelas final, ambang reliable 30 sampel).
   Cek dulu, sesuaikan kalau perlu, JANGAN diubah lagi setelah mulai run.
2. Port logic dari notebook lama ke `src/ingest.py` -> `label.py` ->
   `clean.py` satu per satu. Tiap file sudah ada docstring + referensi ke
   bagian STORY_DEVELOPMENT.md / PLAN_V6_DATA_FIX.md yang jadi sumber logic.
3. `src/split.py`, lalu `src/train.py`, `src/evaluate.py`.
4. Jalankan 3 eksperimen wajib dulu (lihat matriks di rancangan):
   - data-v1 x IndoBERT Large
   - data-v3 x IndoBERT Base
   - data-v3 x IndoBERT Large
   Training berat -> jalanin di Kaggle (git clone repo ini, internet ON).
   Ingest/label/clean/split -> cukup lokal, CPU-only, ringan.
5. `python -m reports.generate_report` -- generate tabel dari
   `experiments/runs.jsonl`, bukan ketik manual lagi.

## Status implementasi

- [x] `config/experiment.yaml`
- [ ] `src/ingest.py`
- [ ] `src/label.py`
- [ ] `src/clean.py`
- [ ] `src/split.py`
- [ ] `src/train.py`
- [ ] `src/evaluate.py`
- [ ] `src/pipeline.py`
- [ ] `tests/`
- [ ] `reports/generate_report.py`
