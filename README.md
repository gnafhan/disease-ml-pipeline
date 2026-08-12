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
- [x] `src/train.py` -- **smoke-test SUDAH lolos di Kaggle GPU (semua 6 kombinasi)**;
      training PENUH (tanpa `--smoke-test`) belum pernah dituntaskan sampai
      selesai -- lihat bagian Training buat status terbaru
- [x] `src/evaluate.py` -- kode lengkap, teruji dgn data sintetis
- [x] `src/run_all_experiments.py` -- entrypoint Kaggle, orkestrasi 6 kombinasi
      teruji (training di-mock di test, tapi orkestrasinya sendiri sudah
      jalan sungguhan di Kaggle); `--push-git` backup progress ke GitHub tiap
      kombinasi, `--push-hf` auto-push model terbaik ke HuggingFace Hub di akhir
- [x] `src/push_to_hf.py` -- pilih run non-smoke-test dgn `test_f1_macro_reliable_only`
      tertinggi, push ke HuggingFace Hub + generate model card otomatis
- [x] `tests/` -- 47 test (label/clean/evaluate/split/run_all_experiments/
      generate_report/train_config/push_to_hf: unit test murni, tanpa
      GPU/internet; ingest: jalan terhadap raw_data asli). Jalankan:
      `source ../.venv/bin/activate && pytest tests/ -v`
- [x] `reports/generate_report.py` -- sudah jalan, generate kerangka + tabel;
      dedup otomatis per run_id (entri terbaru menang) supaya hasil smoke-test
      lama nggak numpuk di matriks final

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

## Training -- BUTUH GPU, smoke-test sudah lolos di Kaggle, training penuh belum kelar

`src/train.py` sudah lengkap (focal loss, class weights, symptom-flag
injection, dropout-after-load fix dari V5, semuanya di-port dari
`v5_fixed.py`). Bridge Claude (3.8GB RAM, no GPU, no internet ke HuggingFace)
dan Mac langsung (CPU-only) keduanya nggak cukup buat fine-tune IndoBERT
dalam waktu wajar, jadi verifikasi sungguhan HARUS di Kaggle GPU.

Status per sesi terakhir: `--smoke-test` (1 epoch, 2 sampel/kelas) SUDAH
lolos 6/6 kombinasi di Kaggle T4 tanpa error -- ini juga sekaligus
membuktikan bagian yang sebelumnya nggak bisa diverifikasi dari sandbox
manapun (download model pretrained dari HuggingFace, tokenizer, GPU
fine-tuning loop) beneran jalan. Training PENUH (tanpa `--smoke-test`, 12
epoch x 6 kombinasi) belum pernah dituntaskan sampai selesai -- itu yang
seharusnya jadi hasil akhir buat matriks perbandingan di BAB IV laporan TA.

Bug yang udah ketemu & dibenerin lewat proses smoke-test-di-Kaggle ini (baik
`pytest` di sandbox maupun unit test murni nggak bisa nangkep ini karena
keduanya nggak pernah benar-benar memanggil kode training asli dgn config
asli):
- `train.py` baca `cfg["split"]["seed"]` padahal `seed` itu top-level di
  config (`cfg["seed"]`) -- bikin SEMUA 6 kombinasi gagal serentak dgn
  `KeyError`. Sudah ada regression test (`tests/test_train_config.py`) yang
  nge-cek tiap akses `cfg[...]` di `train.py` beneran match struktur YAML.
- `reports/generate_report.py` belum dedup per `run_id` -- kalau
  smoke-test lalu training asli jalan berurutan, `runs.jsonl` numpuk 2 entri
  per `run_id` dan matriks akhir jadi campur. Sudah dibenerin (entri
  TERAKHIR per `run_id` yang dipakai), diuji di `tests/test_generate_report.py`.

Yang SUDAH diverifikasi jalan tanpa error lewat unit test murni (nggak butuh
GPU/internet): urutan 6 kombinasi, lanjut otomatis walau 1 gagal, filter
`--only`, resume via `--skip-existing`, `--push-git` (backup progress ke
GitHub tiap kombinasi), `--push-hf` (pilih & push model terbaik ke
HuggingFace Hub). Juga: focal loss (loss makin besar kalau prediksi makin
salah), symptom-flag extraction (flag `[CAMPAK]` udah gak pernah muncul
lagi), dan `build_input` (aman kalau usia/gender kosong).

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

### 4. Simpan SEMUA credential sebagai Kaggle Secret (satu kali, bukan di kode)

Baik token GitHub maupun HuggingFace SELALU lewat Kaggle Secrets, TIDAK
pernah ditulis langsung di cell manapun -- jadi walau kamu ganti/hapus/bikin
ulang notebook, atau training ulang berkali-kali, kamu nggak pernah perlu
edit kode buat masukin credential lagi. Bahkan nama repo GitHub/HuggingFace
kamu juga disimpan sbg Secret (bukan cuma token-nya), supaya Cell 5 nggak
ada satupun bagian yang perlu diganti manual.

Cara kerja Kaggle Secrets: nilai Secret (token, dsb) disimpan SEKALI di
level akun Kaggle kamu (Add-ons -> Secrets -> Add Secret). Notebook BARU
yang kamu bikin nanti tetap perlu "attach" Secret itu lewat menu yang sama
(centang nama Secret-nya di notebook itu) -- tapi ini cuma klik toggle, kamu
TIDAK ketik ulang nilai token-nya. Sekali attach, Cell 1 di bawah otomatis
bisa baca semuanya.

Secret yang perlu dibuat (Add-ons -> Secrets -> Add Secret, di notebook Kaggle):

| Nama Secret | Isi | Wajib? |
|---|---|---|
| `GITHUB_TOKEN` | Personal access token GitHub (Settings -> Developer settings -> Personal access tokens -> Fine-grained token). Read-only cukup kalau cuma buat clone repo private; kalau mau `--push-git`, HARUS scope write (Contents: Read and write). | Wajib kalau repo private ATAU mau `--push-git` |
| `GITHUB_REPO` | `<username>/<nama-repo>`, mis. `gnafhan/disease-ml-pipeline` | Wajib kalau mau `--push-git` |
| `HF_TOKEN` | User Access Token HuggingFace (huggingface.co -> Settings -> Access Tokens -> New token), scope **Write** | Wajib kalau mau `--push-hf` |
| `HF_REPO_ID` | `<username>/<nama-model>`, mis. `gnafhan/pkt-indobert-best` | Wajib kalau mau `--push-hf` |

Nggak ada satupun dari ini yang perlu username asli akun kamu selain di
`GITHUB_REPO`/`HF_REPO_ID` (itu nama REPO, bukan credential) -- token
GitHub dan HuggingFace keduanya cukup dipakai sendirian tanpa perlu username
terpisah, keduanya nggak validasi token itu harus cocok sama username tertentu.

### 5. Isi cell notebook, urut dari atas

```python
# Cell 1 -- SATU-SATUNYA tempat credential "masuk". Semua nilai narik dari
# Kaggle Secrets (Add-ons -> Secrets), nggak ada token/nama-repo yang
# ditulis langsung di sini -- cell ini nggak pernah perlu diedit lagi
# walau kamu ganti akun/repo, TINGGAL ganti isi Secret-nya di menu Kaggle.
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()


def _try_secret(name):
    try:
        return secrets.get_secret(name)
    except Exception:
        return None


for name in ["GITHUB_TOKEN", "GITHUB_REPO", "HF_TOKEN", "HF_REPO_ID"]:
    value = _try_secret(name)
    if value:
        os.environ[name] = value

github_token = os.environ.get("GITHUB_TOKEN")
github_repo = os.environ.get("GITHUB_REPO")
if github_token and github_repo:
    repo_url = f"https://{github_token}@github.com/{github_repo}.git"
elif github_repo:
    repo_url = f"https://github.com/{github_repo}.git"  # repo public, token nggak wajib buat clone
else:
    raise SystemExit("Secret GITHUB_REPO belum ke-attach di notebook ini -- "
                      "Add-ons -> Secrets, centang GITHUB_REPO (dan GITHUB_TOKEN kalau perlu).")

!git clone {repo_url} repo
%cd repo
```

```python
# Cell 2 -- install dependency (torch/transformers biasanya udah ada di image
# Kaggle, tapi jalankan tetap buat mastiin versi cocok)
!pip install -q -r requirements.txt
```

```python
# Cell 3 -- salin data yang sudah diproses dari Kaggle Dataset. Path Kaggle
# Input kadang beda format (flat vs nested per-owner), jadi dicari otomatis.
import glob, os, shutil

matches = [m for m in glob.glob("/kaggle/input/**/pkt-processed-data", recursive=True) if os.path.isdir(m)]
assert matches, f"pkt-processed-data tidak ketemu. Isi /kaggle/input: {os.listdir('/kaggle/input')}"
src = matches[0]
print("Pakai sumber data:", src)

os.makedirs("data", exist_ok=True)
shutil.copytree(src, "data/processed", dirs_exist_ok=True)
!ls data/processed  # harus kelihatan v1/ v2/ v3/
```

```python
# Cell 4 -- smoke test dulu (cepat, ~1-2 menit) sebelum run penuh
!python -m src.run_all_experiments --smoke-test
```

```python
# Cell 5 -- run sungguhan, semua 6 kombinasi. Command-nya dibangun dari env
# var (hasil Cell 1), BUKAN hardcode -- jadi cell ini juga nggak pernah
# perlu diedit walau nama repo GitHub/HuggingFace kamu ganti, tinggal ganti
# Secret-nya di menu Kaggle.
#   --push-git : backup experiments/runs.jsonl ke GitHub setelah TIAP
#                kombinasi (bukan nunggu ke-6 kelar). Otomatis SKIP kalau
#                GITHUB_TOKEN/GITHUB_REPO nggak ke-attach, bukan error fatal.
#   --push-hf  : setelah SEMUA kombinasi selesai, otomatis push model dgn
#                test_f1_macro_reliable_only TERTINGGI ke HuggingFace Hub.
#                Cuma dipasang kalau Secret HF_REPO_ID ke-attach.
cmd = "python -m src.run_all_experiments --push-git"
if os.environ.get("HF_REPO_ID"):
    cmd += f" --push-hf {os.environ['HF_REPO_ID']}"
print("Menjalankan:", cmd)
!{cmd}
```

Kalau sesi Kaggle mati/keputus di tengah (limit ~9-12 jam per sesi, kuota GPU
mingguan habis, atau kernel crash), buka notebook baru (Secret yang sudah
di-attach otomatis ikut, nggak perlu attach ulang selama masih notebook yang
sama), ulangi Cell 1-3 (clone ulang -- Cell 1 otomatis bawa `runs.jsonl`
terbaru yang sebelumnya di-push `--push-git`), lalu Cell 5 tapi tambah
`--skip-existing`:
```python
# Cell 5 (versi resume) -- sama kayak Cell 5 biasa, tambah --skip-existing
cmd = "python -m src.run_all_experiments --push-git --skip-existing"
if os.environ.get("HF_REPO_ID"):
    cmd += f" --push-hf {os.environ['HF_REPO_ID']}"
print("Menjalankan:", cmd)
!{cmd}
```
Ini bakal ngelewatin kombinasi yang udah kelar (yang run_id-nya ada di
`runs.jsonl` hasil clone), tinggal lanjut dari yang belum. Tanpa
`--push-git` di run sebelumnya, langkah ini nggak akan ada gunanya karena
`runs.jsonl` sesi lama nggak pernah sempat ke-backup.

### 6. Bawa hasil balik ke repo lokal

Kalau kamu udah pakai `--push-git`, `runs.jsonl` di GitHub udah paling baru
tiap saat -- tinggal `git pull` biasa dari Mac. Kalau nggak pakai
`--push-git` (mis. run sebentar aja, nggak khawatir sesi mati), di akhir
Cell 5 cetak isi `experiments/runs.jsonl` dan `reports/matriks_perbandingan.md`,
lalu:
```python
# Cell 6 -- commit hasil balik ke GitHub langsung dari Kaggle. Tetap pakai
# env var dari Cell 1, bukan token/repo yang ditulis ulang di sini.
!git config user.email "kaggle-runner@example.com"
!git config user.name "Kaggle Runner"
!git add experiments/runs.jsonl reports/*.md
!git commit -m "results: 6 run selesai dari Kaggle"
!git push https://{os.environ['GITHUB_TOKEN']}@github.com/{os.environ['GITHUB_REPO']}.git master
```
Atau kalau lebih simpel: klik kanan `experiments/runs.jsonl` di file browser
Kaggle -> Download, lalu taruh manual ke `pipeline/experiments/runs.jsonl` di
laptop dan `git pull` biasa dari terminal Mac.

### 7. (Opsional) Push model terbaik ke HuggingFace Hub

Kalau Secret `HF_REPO_ID` ke-attach, ini otomatis kejadian di akhir Cell 5
tanpa langkah tambahan -- model dgn `test_f1_macro_reliable_only` tertinggi
(run smoke-test selalu diabaikan) langsung ke-upload ke
`https://huggingface.co/<HF_REPO_ID>` lengkap dengan model card (metrik +
data version + disclaimer batasan model).

Kalau lupa attach `HF_REPO_ID` waktu run Cell 5, atau mau push run TERTENTU
(bukan otomatis yang skornya tertinggi), bisa dipanggil manual belakangan
selama `experiments/<run_id>/` masih ada di working directory sesi yang sama
(begitu sesi Kaggle-nya mati, folder model ini HILANG -- makanya kalau mau
model tersimpan permanen, push-nya harus sebelum sesi itu berakhir):
```python
# Cell 7 -- push manual, tetap pakai Secret HF_REPO_ID (bukan hardcode)
!python -m src.push_to_hf --repo-id {os.environ["HF_REPO_ID"]}
# atau push run tertentu, bukan otomatis yang terbaik:
!python -m src.push_to_hf --repo-id {os.environ["HF_REPO_ID"]} --run-id v3_large
```

Model yang di-push cuma model FINAL (bobot terbaik hasil `load_best_model_at_end`),
bukan tiap checkpoint per-epoch -- itu memang sengaja disaring biar repo
HuggingFace-nya nggak penuh sampah checkpoint yang nggak kepakai.
