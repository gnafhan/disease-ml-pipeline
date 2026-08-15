# Rencana Eksperimen Optimasi Performa Inferensi IndoBERT

## 1. Keputusan eksperimen

Eksperimen ini menguji cara menjalankan **satu model klasifikasi yang sudah
selesai dilatih** pada server. Model, label, preprocessing, test set, dan
panjang input dibekukan. Variabel yang diubah hanya runtime dan representasi
numerik model.

Judul kerja:

> **Optimasi Performa Inferensi Model IndoBERT untuk Klasifikasi Anamnesis pada
> Server Menggunakan ONNX Runtime dan Dynamic INT8 Quantization**

Model kandidat utama adalah private repository
`gnafhan/pkt-ta-indobert-v4-large`. Model ini dipilih karena menghasilkan macro
F1 seluruh 12 kelas tertinggi pada eksperimen V4, yaitu 0,5944. Pemilihan model
tersebut adalah prasyarat eksperimen, bukan variabel yang ikut dibandingkan.

Pertanyaan utama:

> Seberapa besar ONNX Runtime dan dynamic INT8 quantization mengubah latency,
> throughput, penggunaan sumber daya, serta kualitas prediksi IndoBERT pada
> server yang sama?

Eksperimen tidak bertujuan mencari model dengan akurasi terbaik. Accuracy dan
F1 tetap dihitung untuk memastikan optimasi inferensi tidak merusak kualitas
prediksi model yang dibekukan.

## 2. Scope

### Termasuk

- satu model tetap: V4 large;
- satu server fisik/virtual yang sama;
- API inference berbasis FastAPI;
- baseline PyTorch FP32;
- ONNX Runtime FP32;
- ONNX Runtime dynamic INT8;
- pengujian kualitas prediksi pada test set tetap;
- micro-benchmark berdasarkan panjang input;
- load test dan stress test HTTP;
- pencatatan latency, throughput, RAM, CPU, ukuran artefak, dan error rate;
- pemilihan konfigurasi deployment berdasarkan aturan yang ditetapkan sebelum
  hasil eksperimen diketahui.

### Tidak termasuk

- retraining atau hyperparameter tuning;
- perbandingan IndoBERT base dengan large;
- perubahan cleaning atau split data;
- MLOps end-to-end, auto-retraining, dan model monitoring produksi;
- Kubernetes, autoscaling, atau multi-node serving;
- TensorRT/OpenVINO pada eksperimen utama;
- kalibrasi confidence;
- klaim diagnosis atau validasi klinis.

## 3. Hipotesis engineering

| ID | Hipotesis |
|---|---|
| H1 | ONNX Runtime FP32 menurunkan latency p95 dibanding PyTorch FP32 tanpa perubahan material pada macro F1. |
| H2 | ONNX Runtime INT8 mengurangi ukuran artefak dan peak RAM dibanding ONNX FP32. |
| H3 | ONNX Runtime INT8 meningkatkan throughput dibanding PyTorch FP32 dengan penurunan macro F1 tidak lebih dari 0,01. |
| H4 | Keuntungan runtime berbeda menurut panjang input dan tingkat concurrency. |

Hasil negatif tetap valid. Jika INT8 lebih lambat pada CPU server atau menurunkan
macro F1 melewati batas, konfigurasi tersebut tidak dipilih untuk deployment.

## 4. Arsitektur

```text
Benchmark client (k6 / harness Python)
                |
                v
        POST /v1/predict
                |
                v
         FastAPI + Uvicorn
                |
     +----------+----------+
     |          |          |
 PyTorch     ONNX FP32   ONNX INT8
   FP32         CPU         CPU
     |          |          |
     +----------+----------+
                |
                v
  label + score + model/runtime metadata
```

Setiap eksperimen menjalankan hanya satu backend. Seluruh backend memakai API
contract, tokenizer, preprocessing, `max_length`, dan postprocessing yang sama.

## 5. Stack yang digunakan

| Lapisan | Teknologi | Alasan |
|---|---|---|
| Bahasa | Python 3.11 atau versi yang didukung server | Kompatibel dengan PyTorch, Transformers, dan ONNX Runtime. |
| API | FastAPI + Pydantic | Contract request/response eksplisit dan mudah diuji. |
| ASGI server | Uvicorn, satu worker | Menghindari perbedaan akibat replikasi model antarproses. |
| Baseline | PyTorch + Hugging Face Transformers | Runtime asli model. |
| Export | Hugging Face Optimum/ONNX exporter | Menjaga export transformer konsisten. |
| Runtime optimasi | ONNX Runtime CPU Execution Provider | Target utama serving CPU. |
| Quantization | ONNX Runtime dynamic INT8 | Metode yang sesuai untuk transformer pada eksperimen CPU. |
| Offline metrics | scikit-learn | Accuracy, macro F1, weighted F1, recall per kelas. |
| Load generator | k6 | Concurrent HTTP load dan percentile latency. |
| Resource sampler | psutil | RSS memory dan utilisasi CPU proses API. |
| Packaging | Docker Compose, bila tersedia | Menyamakan dependency dan resource limit antarbackend. |
| Output | CSV + JSON + Markdown | Mudah diaudit dan dipakai dalam laporan TA. |

Versi dependency harus dikunci setelah audit server. Jangan memasang
`onnxruntime` dan `onnxruntime-gpu` bersamaan pada environment eksperimen CPU.

## 6. Gate 0: inventaris server

Eksperimen belum boleh dimulai sebelum tabel berikut terisi. Hardware dan
software server memengaruhi hasil quantization; hasil dari mesin lain tidak
bisa dianggap berlaku otomatis.

| Properti | Nilai aktual |
|---|---|
| Vendor/model server | _belum diisi_ |
| OS dan kernel | _belum diisi_ |
| CPU | _belum diisi_ |
| Arsitektur CPU | _belum diisi_ |
| Physical/logical core | _belum diisi_ |
| Instruction set relevan (AVX2/AVX-512/VNNI) | _belum diisi_ |
| RAM total | _belum diisi_ |
| GPU dan VRAM, bila ada | _belum diisi_ |
| Storage | _belum diisi_ |
| Python | _belum diisi_ |
| PyTorch | _belum diisi_ |
| Transformers | _belum diisi_ |
| ONNX | _belum diisi_ |
| ONNX Runtime | _belum diisi_ |
| Docker | _belum diisi_ |
| Commit/revision aplikasi | _belum diisi_ |

Rencana utama mengasumsikan CPU. Jika server mempunyai GPU NVIDIA, eksperimen
CPU tetap dapat dijalankan sebagai scope utama. TensorRT/FP16 hanya boleh
ditambahkan sebagai eksperimen lanjutan setelah matriks utama selesai; ia tidak
boleh mengganti definisi eksperimen di tengah pengukuran.

## 7. Artefak model dan kontrol versi

Sebelum konversi, simpan identitas artefak berikut:

| Artefak | Nilai |
|---|---|
| Hugging Face repo | `gnafhan/pkt-ta-indobert-v4-large` |
| Revision/commit model | _belum diisi_ |
| SHA-256 model FP32 | _belum diisi_ |
| SHA-256 tokenizer | _belum diisi_ |
| Jumlah label | 12 |
| `max_length` | 192 |
| Test-set SHA-256 | _belum diisi_ |
| Input builder revision | _belum diisi_ |

Token Hugging Face hanya dibaca dari environment/secret manager. Token tidak
boleh dimasukkan ke image Docker, konfigurasi, Git, log, atau report.

## 8. Konfigurasi runtime

| ID | Backend | Model format | Presisi/bobot | Execution provider | Peran |
|---|---|---|---|---|---|
| B0 | PyTorch eager | Safetensors/PyTorch | FP32 | CPU | Baseline |
| B1 | ONNX Runtime | ONNX | FP32 | CPU | Uji pengaruh runtime/graph optimization |
| B2 | ONNX Runtime | ONNX quantized | Dynamic INT8 | CPU | Uji pengaruh quantization |

Konfigurasi yang dikontrol sama untuk ketiganya:

| Variabel kontrol | Nilai awal |
|---|---|
| Model | V4 large revision yang dibekukan |
| Tokenizer | Tokenizer model yang sama |
| Max sequence length | 192 token |
| Truncation/padding | Identik untuk semua backend |
| API schema | Identik |
| Uvicorn worker | 1 |
| Server host | Mesin yang sama |
| Test set | Split V4 yang sama |
| Randomness | Inference/evaluation mode, dropout nonaktif |
| Request logging | Body anamnesis tidak dicatat |
| Background workload | Dihentikan atau dicatat |

Jumlah thread PyTorch dan ONNX harus dicatat. Nilai awal disamakan dengan
jumlah core fisik atau hasil preflight yang ditetapkan sebelum benchmark. Thread
tuning tidak ikut menjadi variabel eksperimen utama.

## 9. Contoh konfigurasi aplikasi

```yaml
model:
  repo_id: gnafhan/pkt-ta-indobert-v4-large
  revision: "<frozen-hf-commit>"
  max_length: 192

runtime:
  backend: pytorch  # pytorch | onnx_fp32 | onnx_int8
  device: cpu
  intra_op_threads: "<ditetapkan-setelah-audit-server>"

server:
  host: 0.0.0.0
  port: 8000
  workers: 1
  log_request_body: false

benchmark:
  warmup_requests: 20
  measured_requests: 100
  repeats: 3
  timeout_seconds: 30
```

File konfigurasi per backend boleh berbeda hanya pada `runtime.backend` dan
lokasi artefak model.

## 10. API contract

### `GET /healthz`

Memastikan proses hidup dan model telah dimuat.

```json
{
  "status": "ready",
  "model_version": "v4-large@<revision>",
  "runtime": "onnx-int8"
}
```

### `POST /v1/predict`

Request mengikuti input yang dipakai saat training:

```json
{
  "anamnesa": "demam tiga hari, mual, muntah",
  "age_years": 20,
  "sex": "P",
  "bulan_kunjung": 8
}
```

Response:

```json
{
  "prediction": "Suspek Dengue",
  "score": 0.87,
  "model_version": "v4-large@<revision>",
  "runtime": "onnx-int8",
  "latency_ms": 123.45
}
```

Field `score` adalah softmax score, bukan probabilitas klinis terkalibrasi.
Request body, token, nomor rekam medis, dan anamnesis tidak boleh disimpan pada
access log atau application log.

Validasi API minimal:

- `anamnesa` wajib dan tidak boleh kosong;
- `age_years` berada pada 0–110 jika diisi;
- `sex` mengikuti nilai yang dipakai model;
- `bulan_kunjung` berada pada 1–12 jika diisi;
- payload terlalu besar ditolak;
- error response tidak mengembalikan stack trace atau secret.

## 11. Validasi konversi sebelum benchmark

Benchmark hanya berjalan jika B1 dan B2 dapat memproses seluruh test set tanpa
error.

| Pemeriksaan | B0 | B1 | B2 |
|---|---:|---:|---:|
| Seluruh sampel selesai | _ | _ | _ |
| Output shape `(n, 12)` | _ | _ | _ |
| NaN/Inf pada logits | _ | _ | _ |
| Prediction agreement terhadap B0 | 100% | _ | _ |
| Maximum absolute logit difference | 0 | _ | _ |
| Macro F1 | _ | _ | _ |
| Delta macro F1 terhadap B0 | 0 | _ | _ |

B1 seharusnya mendekati baseline secara numerik. B2 boleh berbeda karena
aproksimasi INT8, tetapi tetap harus memenuhi quality gate pada Bagian 18.

## 12. Dataset evaluasi kualitas model

Kualitas B0, B1, dan B2 dihitung pada test set yang sama. Urutan baris,
preprocessing, truncation, mapping label, dan postprocessing harus identik.

Metrik yang dihitung:

- accuracy;
- macro F1 seluruh 12 kelas;
- weighted F1;
- precision, recall, F1, dan support per kelas;
- prediction agreement terhadap B0;
- jumlah prediksi yang berubah;
- delta macro F1 dan delta accuracy terhadap B0.

`F1max` tidak digunakan karena task ini merupakan multiclass single-label dengan
prediksi argmax, bukan pencarian threshold biner/multilabel.

### Tabel hasil kualitas model

| Konfigurasi | Accuracy | Macro F1 | Weighted F1 | Agreement vs B0 | Prediksi berubah | Delta accuracy | Delta macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 PyTorch FP32 | _ | _ | _ | 100% | 0 | 0 | 0 |
| B1 ONNX FP32 | _ | _ | _ | _ | _ | _ | _ |
| B2 ONNX INT8 | _ | _ | _ | _ | _ | _ | _ |

### Tabel performa per kelas

| Kelas | Support test | B0 F1 | B1 F1 | B2 F1 | Delta B1–B0 | Delta B2–B0 |
|---|---:|---:|---:|---:|---:|---:|
| Acute Flaccid Paralysis | _ | _ | _ | _ | _ | _ |
| COVID-19 Konfirmasi | _ | _ | _ | _ | _ | _ |
| Diare Akut | _ | _ | _ | _ | _ | _ |
| Diare Berdarah | _ | _ | _ | _ | _ | _ |
| GHPR | _ | _ | _ | _ | _ | _ |
| Pneumonia/ISPA | _ | _ | _ | _ | _ | _ |
| Sindrom Jaundice Akut | _ | _ | _ | _ | _ | _ |
| Suspek Dengue | _ | _ | _ | _ | _ | _ |
| Suspek HFMD | _ | _ | _ | _ | _ | _ |
| Suspek Leptospirosis | _ | _ | _ | _ | _ | _ |
| Suspek Meningitis/Ensefalitis | _ | _ | _ | _ | _ | _ |
| Suspek Tetanus | _ | _ | _ | _ | _ | _ |

Kelas dengan support kecil tetap dilaporkan, tetapi perubahan satu prediksi
tidak boleh ditafsirkan sebagai bukti stabil tanpa menyebut support-nya.

## 13. Korpus benchmark performa

Benchmark latency memakai tiga bucket panjang teks yang sudah digunakan pada
audit data:

| Bucket | Definisi | Target sampel unik |
|---|---|---:|
| Pendek | ≤15 kata | 30 |
| Sedang | 16–50 kata | 30 |
| Panjang | >50 kata | 30 |

Gunakan 30 payload unik per bucket secara round-robin. Jangan mengirim satu teks
yang sama 100 kali karena hasilnya kurang mewakili variasi tokenisasi. Payload
berasal dari test set private di lingkungan server atau dataset sintetis yang
disetujui; teks mentah tidak ditulis ke output benchmark.

Setiap file workload hanya memuat ID pseudonim, bucket panjang, dan payload yang
tetap berada di server. Report publik hanya menyimpan statistik agregat.

## 14. Protokol micro-benchmark

Tujuan micro-benchmark adalah mengisolasi pengaruh runtime dan panjang input
pada satu request.

Untuk setiap kombinasi backend × bucket:

1. mulai service dengan satu worker;
2. tunggu `/healthz` berstatus `ready`;
3. kirim 20 warm-up request yang tidak dicatat;
4. kirim 100 measured request dengan concurrency 1;
5. ulangi pengukuran tiga kali;
6. restart service sebelum berpindah backend;
7. simpan raw timing non-sensitif ke CSV;
8. laporkan median dari tiga run dan seluruh percentile latency.

### Matriks micro-benchmark

| Backend | Bucket | Warm-up | Request/run | Replikasi | Total terukur | Avg ms | p50 ms | p95 ms | p99 ms | Peak RSS MB | Error rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Pendek | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B0 | Sedang | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B0 | Panjang | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B1 | Pendek | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B1 | Sedang | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B1 | Panjang | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B2 | Pendek | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B2 | Sedang | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |
| B2 | Panjang | 20 | 100 | 3 | 300 | _ | _ | _ | _ | _ | _ |

## 15. Load test dan stress test

Stress test mengukur kemampuan API ketika request datang bersamaan. Gunakan
payload bucket sedang karena ia paling mewakili penggunaan umum dan menghindari
perubahan dua variabel sekaligus. Tiga puluh payload sedang diputar agar tidak
ada keuntungan akibat satu input berulang.

### Definisi profil beban

| Profil | Virtual users | Iterasi per user | Total request/run | Replikasi | Tujuan |
|---|---:|---:|---:|---:|---|
| Normal | 1 | 100 | 100 | 3 | Latency satu pengguna |
| Load | 5 | 100 | 500 | 3 | Beban bersamaan ringan |
| Stress | 20 | 100 | 2.000 | 3 | Antrean, throughput, dan error saat beban tinggi |

Gunakan mode `per-vu-iterations` pada k6 agar jumlah request eksak. Terapkan
timeout yang sama untuk seluruh backend. Sebelum setiap profil, jalankan 20
warm-up request dan pastikan resource server kembali ke kondisi idle yang
ditentukan.

### Matriks hasil load/stress

| Backend | Profil | Request/run | Replikasi | Success | Failed | Error rate | Throughput req/s | p50 ms | p95 ms | p99 ms | Peak RSS MB | Avg CPU % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Normal | 100 | 3 | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| B0 | Load | 500 | 3 | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| B0 | Stress | 2.000 | 3 | _ | _ | _ | _ | _ | _ | _ | _ |
| B1 | Normal | 100 | 3 | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| B1 | Load | 500 | 3 | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| B1 | Stress | 2.000 | 3 | _ | _ | _ | _ | _ | _ | _ | _ |
| B2 | Normal | 100 | 3 | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| B2 | Load | 500 | 3 | _ | _ | _ | _ | _ | _ | _ | _ |
| B2 | Stress | 2.000 | 3 | _ | _ | _ | _ | _ | _ | _ | _ |

Jumlah request merupakan variabel beban; success, failed, error rate,
throughput, dan percentile latency merupakan hasil yang wajib dilaporkan.

## 16. Waktu startup dan ukuran artefak

Waktu startup diukur dari proses API mulai sampai `/healthz` pertama kali
menjawab `ready`. Lakukan lima kali restart per backend. Jangan mencampur startup
time dengan latency request setelah warm-up.

| Backend | Ukuran artefak MB | Startup median ms | Startup p95 ms | Idle RSS MB | Peak RSS MB |
|---|---:|---:|---:|---:|---:|
| B0 PyTorch FP32 | _ | _ | _ | _ | _ |
| B1 ONNX FP32 | _ | _ | _ | _ | _ |
| B2 ONNX INT8 | _ | _ | _ | _ | _ |

## 17. Rumus analisis

Gunakan B0 sebagai denominator tetap.

```text
latency_reduction_pct
= (p95_B0 - p95_variant) / p95_B0 × 100%

speedup
= p95_B0 / p95_variant

throughput_gain_pct
= (throughput_variant - throughput_B0) / throughput_B0 × 100%

memory_reduction_pct
= (peak_RSS_B0 - peak_RSS_variant) / peak_RSS_B0 × 100%

size_reduction_pct
= (size_B0 - size_variant) / size_B0 × 100%

delta_macro_f1
= macro_f1_variant - macro_f1_B0

prediction_agreement
= jumlah_prediksi_sama_dengan_B0 / jumlah_test × 100%
```

Gunakan p95 sebagai metrik latency utama karena rata-rata dapat menyembunyikan
request lambat. p50, p99, dan rata-rata tetap dilaporkan sebagai metrik
sekunder.

## 18. Quality gate dan aturan pemilihan

Threshold berikut adalah keputusan engineering eksperimen, bukan nilai
universal. Ia harus disetujui sebelum hasil benchmark dibuka.

### Gate wajib

| Gate | B1 ONNX FP32 | B2 ONNX INT8 |
|---|---:|---:|
| Seluruh test set berhasil diproses | Wajib | Wajib |
| NaN/Inf logits | 0 | 0 |
| Delta macro F1 terhadap B0 | ≥ -0,001 | ≥ -0,010 |
| Error rate profil normal | 0% | 0% |
| Error rate profil load | 0% | 0% |
| API contract sama | Wajib | Wajib |

### Aturan keputusan

1. keluarkan konfigurasi yang gagal gate kualitas atau functional test;
2. dari konfigurasi yang tersisa, pilih latency p95 terendah pada profil load;
3. gunakan throughput, peak RSS, dan ukuran model sebagai tie-breaker;
4. jika tidak ada varian yang lolos, pertahankan B0;
5. tetap laporkan semua hasil, termasuk varian yang gagal.

Target improvement 20% boleh dipakai sebagai indikator manfaat praktis, tetapi
tidak sebagai alasan menghapus hasil di bawah target.

### Tabel keputusan akhir

| Backend | Quality gate | p95 load | Speedup | Throughput | Peak RSS | Ukuran | Delta F1 | Keputusan |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | Baseline | _ | 1,00× | _ | _ | _ | 0 | _ |
| B1 | _ | _ | _ | _ | _ | _ | _ | _ |
| B2 | _ | _ | _ | _ | _ | _ | _ | _ |

## 19. Kondisi eksperimen yang harus dijaga

- jalankan seluruh backend pada server yang sama;
- gunakan satu Uvicorn worker untuk eksperimen utama;
- jangan menjalankan training, backup, atau workload berat lain bersamaan;
- nonaktifkan dropout dan gunakan inference/evaluation mode;
- gunakan tokenizer dan input builder yang identik;
- gunakan jumlah thread dan resource limit yang sama;
- gunakan client benchmark dan posisi jaringan yang sama;
- catat apakah client berada di localhost, LAN, atau jaringan luar;
- restart service ketika berpindah backend;
- lakukan warm-up sebelum pengukuran;
- simpan timestamp, temperatur CPU jika tersedia, dan kondisi idle;
- jangan memilih hanya run terbaik dari tiga replikasi;
- simpan seluruh raw timing non-sensitif untuk audit.

Jika benchmark client berjalan pada server yang sama, utilisasinya ikut
mengganggu CPU inference. Pilihan yang lebih baik adalah menjalankan k6 dari
mesin client terpisah pada jaringan yang sama. Jika itu tidak tersedia,
localhost tetap boleh dipakai asalkan batasan tersebut dilaporkan dan seluruh
backend diuji dengan kondisi identik.

## 20. Functional dan privacy test

| Test | Hasil yang diharapkan |
|---|---|
| `/healthz` sebelum model siap | Tidak melaporkan `ready` |
| Payload valid | HTTP 200 dan schema response valid |
| Anamnesis kosong | HTTP 422/400 |
| Usia di luar 0–110 | HTTP 422/400 |
| Bulan di luar 1–12 | HTTP 422/400 |
| Payload sangat besar | Ditolak sesuai limit |
| Backend B0/B1/B2 | Mapping label sama dan lengkap |
| Log API | Tidak memuat anamnesis atau token |
| Error internal | Tidak memuat secret atau stack trace ke client |
| Model private | Hanya diunduh dengan secret environment |

## 21. Struktur implementasi yang direncanakan

```text
serving/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── preprocessing.py
│   └── backends/
│       ├── base.py
│       ├── pytorch_backend.py
│       └── onnx_backend.py
├── export/
│   ├── export_onnx.py
│   ├── quantize_int8.py
│   └── validate_parity.py
├── benchmark/
│   ├── workloads/
│   ├── k6_load.js
│   ├── micro_benchmark.py
│   ├── resource_sampler.py
│   └── summarize_results.py
├── config/
│   ├── pytorch-fp32.yaml
│   ├── onnx-fp32.yaml
│   └── onnx-int8.yaml
├── tests/
├── Dockerfile
└── compose.yaml

results/inference/<timestamp>/
├── environment.json
├── artifact_manifest.json
├── quality_metrics.csv
├── per_class_metrics.csv
├── micro_benchmark.csv
├── load_test.csv
├── startup_memory.csv
└── summary.md
```

Artefak model dan payload private tidak boleh dikomit. Hanya konfigurasi,
script, manifest hash, serta hasil agregat non-sensitif yang masuk Git.

## 22. Urutan implementasi server

### Fase 0 — audit server

- isi inventaris hardware/software;
- tentukan CPU execution provider;
- tetapkan jumlah thread dan resource limit;
- pastikan ruang disk cukup untuk model FP32 dan INT8;
- siapkan secret Hugging Face tanpa menuliskannya ke repository.

### Fase 1 — freeze baseline

- download revision model V4 large yang spesifik;
- simpan SHA-256 model/tokenizer/test set;
- implementasikan input builder yang sama dengan training;
- jalankan evaluasi B0 dan simpan metrik baseline.

### Fase 2 — API baseline

- implementasikan FastAPI dan backend interface;
- jalankan B0 PyTorch FP32;
- selesaikan functional/privacy test;
- ukur startup, RAM idle, dan micro-benchmark awal.

### Fase 3 — export dan quantization

- export B1 ONNX FP32;
- validasi shape, logits, prediksi, dan metrik;
- quantize menjadi B2 dynamic INT8;
- ulangi validasi kualitas;
- hentikan proses bila output tidak valid.

### Fase 4 — benchmark terkontrol

- jalankan sembilan sel micro-benchmark;
- jalankan sembilan sel load/stress test;
- ulangi setiap sel tiga kali;
- simpan raw timing dan resource samples;
- jangan mengganti dependency atau konfigurasi di tengah eksperimen.

### Fase 5 — analisis dan deployment

- hitung speedup, penghematan memori, penghematan ukuran, dan delta F1;
- isi seluruh tabel report;
- terapkan quality gate dan aturan keputusan;
- deploy konfigurasi terpilih;
- jalankan smoke test akhir dari mesin client.

## 23. Definition of done

- [ ] spesifikasi server dan versi dependency tercatat;
- [ ] model/revision/test set dibekukan dengan SHA-256;
- [ ] API contract identik untuk tiga backend;
- [ ] B0, B1, dan B2 menyelesaikan evaluasi test set;
- [ ] quality metrics dan per-class metrics tersedia;
- [ ] sembilan skenario micro-benchmark selesai tiga replikasi;
- [ ] sembilan skenario load/stress selesai tiga replikasi;
- [ ] startup time, ukuran artefak, CPU, dan RAM tercatat;
- [ ] tidak ada anamnesis mentah atau secret di log/report;
- [ ] tabel keputusan akhir terisi;
- [ ] konfigurasi terbaik dideploy dan dapat didemokan;
- [ ] hasil negatif tetap dilaporkan;
- [ ] seluruh script dan hasil agregat dapat direproduksi.

## 24. Ancaman validitas

1. **Hardware-specific result.** Keuntungan INT8 bergantung pada instruction set
   CPU. Kesimpulan berlaku pada server uji, bukan semua server.
2. **Network noise.** Latency HTTP memuat waktu jaringan. Posisi client harus
   sama untuk semua backend.
3. **Warm-up dan cache.** Request pertama berbeda dari steady-state. Startup
   dan warmed latency dilaporkan terpisah.
4. **Thermal/background load.** Suhu dan proses lain dapat mengubah hasil.
   Tiga replikasi dan pencatatan kondisi server mengurangi risiko ini.
5. **Small-class metrics.** F1 kelas dengan support test sangat kecil bersifat
   volatil. Support selalu ditampilkan.
6. **Uncalibrated score.** Softmax score tidak diperlakukan sebagai confidence
   klinis.
7. **Private clinical text.** Raw payload tidak boleh keluar dari server atau
   masuk ke log publik.

## 25. Referensi teknis implementasi

- ONNX Runtime documentation: <https://onnxruntime.ai/docs/>
- ONNX Runtime graph optimization:
  <https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html>
- ONNX Runtime quantization:
  <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- FastAPI documentation: <https://fastapi.tiangolo.com/>
- k6 scenarios documentation: <https://grafana.com/docs/k6/latest/using-k6/scenarios/>

Dokumentasi ini menjadi plan pra-implementasi. Nilai hardware, revision model,
dependency, serta hasil tabel diisi setelah akses server tersedia dan sebelum
konfigurasi terbaik ditentukan.
