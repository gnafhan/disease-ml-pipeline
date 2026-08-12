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
- [x] `src/train.py` -- kode lengkap, **BELUM PERNAH DIJALANKAN dgn model sungguhan** (butuh GPU + akses HuggingFace, lihat bagian Training)
- [x] `src/evaluate.py` -- kode lengkap, teruji dgn data sintetis
- [x] `src/run_all_experiments.py` -- entrypoint Kaggle, orkestrasi 6 kombinasi teruji (training di-mock)
- [x] `tests/` -- 26 test (label/clean/evaluate/run_all_experiments: unit test
      murni; ingest: jalan terhadap raw_data asli). `pytest` tidak ada di
      bridge ini (no internet utk install) -- jalankan manual dari terminal Mac:
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
`v5_fixed.py`) tapi **belum pernah benar-benar dijalankan dengan model
sungguhan** -- bridge Claude (3.8GB RAM, no GPU, no internet ke HuggingFace)
dan Mac langsung (CPU-only) keduanya nggak cukup buat fine-tune IndoBERT
dalam waktu wajar. Yang SUDAH diverifikasi jalan tanpa error (lihat
`tests/test_run_all_experiments.py`, 5 test, training di-mock): urutan 6
kombinasi, lanjut otomatis walau 1 gagal, filter `--only`, resume via
`--skip-existing`. Juga sudah dicek terpisah: focal loss (loss makin besar
kalau prediksi makin salah), symptom-flag extraction (flag `[CAMPAK]` sudah
gak pernah muncul lagi), dan `build_input` (aman kalau usia/gender kosong).
Yang BELUM bisa diverifikasi di sini: proses download model pretrained dari
HuggingFace itu sendiri, karena `huggingface.co` juga gak ke-reach dari
sandbox Claude. Ini murni soal akses jaringan, bukan soal logic training-nya.

### Entrypoint: `src/run_all_experiments.py`

Daripada jalanin `src/train.py` manual 6x, pakai wrapper ini -- dia yang atur
urutan, skip yang udah kelar, lanjut walau ada yang gagal (misal OOM), dan
generate laporan di akhir:

```bash
# jalankan SEMUA 6 kombinasi (data-v1/v2/v3 x model-base/large), urut
python -m src.run_all_experiments

# GPU quota abis / sesi Kaggle keputus di tengah? lanjut dari yang belum kelar
python -m src.run_all_experiments --skip-existing

# cuma mau run kombinasi tertentu
python -m src.run_all_experiments --only v1_large,v3_base

# cek dulu semua kombinasi bisa jalan tanpa error (1 epoch, data subset kecil)
# SEBELUM commit ke run penuh yang makan jam-jaman GPU quota
python -m src.run_all_experiments --smoke-test
```

Tiap kombinasi otomatis ke-log ke `experiments/runs.jsonl` (key `val_*` dan
`test_*` terpisah jelas -- lihat docstring `src/evaluate.py` soal kenapa ini
penting, ini yang bikin bingung di V6-Final dulu), dan begitu semua/sebagian
selesai, `reports/matriks_perbandingan.md` + `reports/per_kelas_<run_id>.md`
otomatis ter-generate ulang (bisa dimatikan dengan `--no-generate-report`
kalau mau generate manual belakangan lewat `python -m reports.generate_report`).

Kalau satu kombinasi gagal (paling sering: OOM di model large), wrapper cetak
ringkasan siapa yang gagal di akhir tanpa menghentikan yang lain -- cek
traceback di output, biasanya solusinya turunkan `batch_size` di
`config/experiment.yaml` (efeknya ke SEMUA run karena config dikunci, jadi
kalau diubah demi 1 kombinasi yang OOM, sebaiknya re-run semua biar hyperparameter tetap konsisten lintas kombinasi).

## Cara run di Kaggle -- tutorial lengkap

**Yang dibutuhkan sebelum mulai:** akun GitHub (buat push kode) dan akun
Kaggle (verifikasi nomor HP dulu di Settings -> Phone Verification, syarat
wajib buat pakai GPU).

### 1. Push kode (bukan data) ke GitHub

```bash
cd pipeline
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin master
```
Boleh private repo. `raw_data/` dan `data/` sudah di `.gitignore` jadi TIDAK
ikut ke-push -- data pasien tetap cuma di laptop kamu, bukan di GitHub.

### 2. Upload data yang sudah diproses sebagai Kaggle Dataset (private)

Data mentah/pasien jangan pernah diupload manapun selain yang memang dibutuhkan
untuk training. Yang perlu diupload ke Kaggle cuma folder `data/processed/`
(sudah dianonim -- No.RM/hash pasien doang, bukan nama/NIK).

- Buka kaggle.com -> Create -> New Dataset.
- Upload folder `data/processed/` (isinya `v1/`, `v2/`, `v3/` masing-masing
  ada `train.csv`, `val.csv`, `test.csv`).
- **Set visibility ke Private.**
- Beri nama, misal `pkt-processed-data`.

### 3. Buat Kaggle Notebook baru

- kaggle.com -> Code -> New Notebook.
- Klik "Add Input" -> cari dataset `pkt-processed-data` yang baru diupload -> Add.
- Panel kanan -> Settings:
  - **Accelerator: GPU T4 x2** (atau P100, tergantung kuota tersisa).
  - **Internet: On.**

### 4. (Kalau repo private) Simpan GitHub token sebagai Kaggle Secret

- Buat token di GitHub: Settings -> Developer settings -> Personal access
  tokens -> Fine-grained token, scope read-only ke repo ini saja.
- Di notebook Kaggle: menu Add-ons -> Secrets -> Add Secret, nama
  `GITHUB_TOKEN`, isi token-nya.

### 5. Isi cell notebook, urut dari atas

```python
# Cell 1 -- clone repo (kalau private, pakai token dari Secrets)
from kaggle_secrets import UserSecretsClient
try:
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    repo_url = f"https://{token}@github.com/<username>/<nama-repo>.git"
except Exception:
    repo_url = "https://github.com/<username>/<nama-repo>.git"  # repo public

!git clone {repo_url} repo
%cd repo
```

```python
# Cell 2 -- install dependency (torch/transformers biasanya udah ada di image
# Kaggle, tapi jalankan tetap buat mastiin versi cocok)
!pip install -q -r requirements.txt
```

```python
# Cell 3 -- salin data yang sudah diproses dari Kaggle Dataset ke lokasi yang
# dibaca src/train.py (ganti "pkt-processed-data" sesuai nama dataset kamu)
!mkdir -p data
!cp -r /kaggle/input/pkt-processed-data/* data/processed/ 2>/dev/null || \
 cp -r /kaggle/input/pkt-processed-data data/processed
!ls data/processed  # harus kelihatan v1/ v2/ v3/
```

```python
# Cell 4 -- smoke test dulu (cepat, ~1-2 menit) sebelum run penuh
!python -m src.run_all_experiments --smoke-test
```

```python
# Cell 5 -- run sungguhan, semua 6 kombinasi
!python -m src.run_all_experiments
```

Kalau sesi Kaggle timeout/keputus di tengah (limit ~9-12 jam per sesi, atau
kuota GPU mingguan habis), buka notebook lagi lain waktu dan jalankan ulang
Cell 5 dengan tambahan `--skip-existing` -- otomatis lanjut dari kombinasi
yang belum kelar, gak perlu ngulang dari 0.

### 6. Bawa hasil balik ke repo lokal

Di akhir Cell 5, cetak isi `experiments/runs.jsonl` dan
`reports/matriks_perbandingan.md`, lalu:
```python
# Cell 6 -- commit hasil balik ke GitHub langsung dari Kaggle
!git config user.email "you@example.com"
!git config user.name "Kaggle Runner"
!git add experiments/runs.jsonl reports/*.md
!git commit -m "results: 6 run selesai dari Kaggle"
!git push https://{token}@github.com/<username>/<nama-repo>.git master
```
Atau kalau lebih simpel: klik kanan `experiments/runs.jsonl` di file browser
Kaggle -> Download, lalu taruh manual ke `pipeline/experiments/runs.jsonl` di
laptop dan `git pull` biasa dari terminal Mac.
