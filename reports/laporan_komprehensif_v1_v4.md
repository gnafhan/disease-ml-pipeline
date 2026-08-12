# Laporan Komprehensif Eksperimen Klasifikasi Sindrom PKT (V1–V4)

## Ringkasan eksekutif

Eksperimen ini mengklasifikasikan anamnesis berbahasa Indonesia ke dalam 12
sindrom surveilans menggunakan IndoBERT. Versi data berkembang dari korpus
berlabel ICD yang belum dibersihkan (V1) menjadi korpus yang diaudit dengan
quality gate dan pemisahan grup pasien/template (V4). Seluruh training penuh
yang dilaporkan di bawah selesai di Kaggle GPU; smoke test tidak dimasukkan.

Hasil akhir menunjukkan dua hal yang harus dibaca bersama. Pada protokol V3,
`v3_base` memberi macro F1 kelas reliable tertinggi, 0,7898. Pada protokol V4
yang lebih ketat, `v4_large` memberi macro F1 seluruh 12 kelas tertinggi,
0,5944, sedangkan `v4_base` memberi akurasi tertinggi, 0,8487. V4 bukan bukti
bahwa model otomatis “lebih baik” daripada V3 karena test split dan ukuran
test set berubah secara material. Kontribusi V4 adalah membuat evaluasi lebih
auditabel: tidak ada baris yang hilang ketika split, tidak ada pasien maupun
template identik yang melintasi subset, dan noise eksplisit dibuang secara
deterministik tanpa memakai label test.

Rekomendasi model operasional untuk tahap ini adalah **V4 large** bila
prioritasnya cakupan 12 kelas dan artefak terunggah ke Hugging Face. Bila biaya
GPU/waktu menjadi batasan, **V4 base** merupakan alternatif yang masuk akal:
akurasi test lebih tinggi 0,0069 dengan waktu training 37,6% dari V4 large,
meskipun macro F1 12 kelas lebih rendah 0,0089. Lima kelas dengan total kurang
dari 30 observasi tetap harus diperlakukan sebagai *insufficient-data classes*,
bukan sebagai bukti model siap dipakai secara klinis.

## Ruang lingkup dan protokol tetap

Korpus terdiri dari dua sumber rumah sakit dan memakai 12 kelas final yang
sama pada seluruh versi: Pneumonia/ISPA, Suspek Dengue, COVID-19 Konfirmasi,
Diare Akut, Acute Flaccid Paralysis, Sindrom Jaundice Akut, Suspek HFMD,
Suspek Tetanus, GHPR, Suspek Leptospirosis, Diare Berdarah, dan Suspek
Meningitis/Ensefalitis. Label direkonstruksi dari mapping ICD yang kemudian
dibandingkan dengan kolom sindrom pada sumber yang menyediakannya. Data mentah
dan teks anamnesis tidak dimasukkan ke repository atau laporan ini.

Seluruh run memakai dua backbone: `indolem/indobert-base-uncased` (base) dan
`indobenchmark/indobert-large-p1` (large). Konfigurasi training yang terkunci
memakai seed 42, panjang input maksimum 192 token, batch size 8 dengan
gradient accumulation 2, learning rate 1e-5, maksimum 12 epoch, early stopping
patience 4, focal loss gamma 2, label smoothing 0,1, dan weight decay 0,05.
Model dipilih berdasarkan macro F1 kelas reliable pada validation set;
metrik headline di bawah selalu dihitung pada test set.

Kelas disebut *reliable* bila jumlah sampel training minimal 30. Dengan aturan
ini ada tujuh kelas reliable dan lima kelas langka. Macro F1 seluruh kelas
tetap dilaporkan agar kelemahan pada kelas langka tidak tersembunyi; macro F1
reliable dipakai sebagai metrik pemilihan agar satu atau dua sampel test tidak
mendominasi keputusan checkpoint.

## Evolusi data

| Versi | Perubahan terhadap versi sebelumnya | Total | Train | Validasi | Test | Kelas | Status evaluasi |
|---|---|---:|---:|---:|---:|---:|---|
| V1 | Ingest dan pelabelan ICD; belum ada penghapusan catatan kontrol atau COVID insidental. | 5.829 | 4.079 | 548 | 499 | 12 | Baseline awal. |
| V2 | Menghapus kunjungan kontrol/post-ranap tanpa keluhan aktif dan COVID-19 insidental pada konteks obstetri/bedah. | 5.249 | 3.673 | 524 | 468 | 12 | Split stratified per baris. |
| V3 | Data sama dengan V2; menambahkan flag `reliable` (ambang ≥30 sampel training) untuk pelaporan dan pemilihan model. Tidak ada baris yang dihapus. | 5.249 | 3.673 | 524 | 468 | 12 | Split stratified per baris; diagnosis leakage dilakukan kemudian. |
| V4 | Adaptive-header ingestion, normalisasi metadata/teks, penghapusan noise ter-audit, deduplikasi kunjungan, pseudonimisasi ID, dan split connected-group pasien/template dengan stratifikasi kelas×sumber. | 4.840 | 3.385 | 728 | 727 | 12 | Protokol final yang lebih ketat. |

V2 mengurangi 580 baris dibanding V1. V3 mempertahankan jumlah tersebut,
karena tujuannya bukan cleaning tambahan tetapi memisahkan interpretasi metrik
kelas dengan dukungan memadai dari kelas langka. V4 berakhir dengan 4.840
baris: 432 baris dihapus oleh quality gate khusus V4 setelah tahap V2, dan
perbaikan deteksi header menaikkan jumlah raw ingestion dari 5.908 menjadi
5.933 baris. Angka V4 karena itu tidak dapat dihitung hanya sebagai 5.249
dikurangi 432.

### Audit perubahan V4

| Keputusan quality gate V4 | Baris | Tujuan |
|---|---:|---|
| Teks anamnesis kosong | 106 | Tidak menyediakan sinyal supervisi teks. |
| Kunjungan pasien+kelas+teks identik berulang | 310 | Menghapus pengulangan observasi yang tidak menambah informasi. |
| Template identik dengan label konflik | 6 | Mencegah satu template generik memetakan ke kelas berbeda. |
| Catatan kontrol tanpa sinyal klinis | 1 | Menghilangkan catatan administratif. |
| Teks sangat pendek tanpa sinyal klinis | 9 | Menghilangkan catatan yang tidak informatif. |
| Dipertahankan | 4.840 | Tetap memakai taksonomi 12 kelas. |

Aturan tersebut tidak memakai keyword untuk mengganti label ICD. Anchor gejala
hanya dipakai sebagai pemeriksaan kualitas/diagnostik; ia tidak memutuskan
kelas baru. V4 juga tidak menghapus kelas langka, tidak melakukan oversampling
ke validation/test, dan tidak menulis nomor rekam medis sumber. ID pada V4
diubah menjadi SHA-256 berbasis sumber dan nomor rekam medis; tindakan ini
adalah pseudonimisasi, bukan anonimisasi, sehingga dataset tetap private.

## Temuan EDA yang memotivasi V4

1. **Kehilangan baris dan leakage template pada V3.** V3 memiliki 5.249 baris
   sebelum split tetapi hanya 4.665 baris yang tertulis ke tiga subset; 584
   baris (11,1%) hilang ketika leak guard dijalankan setelah split. Selain itu,
   12 grup template identik muncul lintas subset. V4 membentuk connected group
   pasien/template sebelum split: 4.840 dari 4.840 baris tertulis, patient
   overlap group = 0, dan template overlap group = 0.

2. **Catatan kosong, duplikasi, dan konflik label.** EDA V3 menemukan 87
   catatan kosong, 449 baris dalam 166 template berulang, serta 91 baris pada
   dua grup template yang memiliki lebih dari satu label. V4 menghapus semua
   template berkonflik dan teks kosong; 84 baris dalam 32 template berulang
   yang tersisa boleh ada, tetapi seluruh anggota template berada di subset
   yang sama.

3. **Keterbatasan kelas langka.** GHPR (25), Suspek Tetanus (28), Suspek
   Leptospirosis (14), Diare Berdarah (14), dan Suspek
   Meningitis/Ensefalitis (8) berada di bawah ambang 30 observasi pada V4.
   Test support per kelas berada pada kisaran 1–4 kasus. Satu prediksi salah
   dapat mengubah F1 secara ekstrem, sehingga angka per kelas ini bersifat
   deskriptif dan tidak cukup untuk klaim kesiapan deployment.

4. **Bias sumber dan tumpang tindih gejala.** COVID-19 Konfirmasi seluruhnya
   berasal dari satu rumah sakit; Pneumonia/ISPA dan Diare Akut masing-masing
   97,4% dan 99,1% dari satu sumber pada V4. Baseline source/visit type saja
   menghasilkan akurasi 0,2242 pada V4, sedangkan metadata klinis saja hanya
   menghasilkan macro F1 reliable 0,0955. Model utama tidak diberi kolom
   sumber, tetapi gaya dokumentasi rumah sakit tetap dapat menjadi shortcut.
   Kesalahan baseline teks paling sering terjadi di antara COVID-19,
   Pneumonia/ISPA, Suspek Dengue, dan Diare Akut—empat kelas yang berbagi
   keluhan demam, batuk, mual/muntah, atau diare.

## Matriks hasil eksperimen

Tabel ini memakai entri **full-training terakhir** untuk setiap `run_id` pada
`experiments/runs.jsonl`; smoke test dan catatan eksperimen lama yang terganti
tidak dihitung. Tanda “—” berarti metrik belum direkam oleh runner legacy,
bukan nilai nol.

| Run | Data | Backbone | Epoch selesai | Akurasi test | Macro F1 12 kelas | Macro F1 reliable | Weighted F1 | Waktu training | Interpretasi |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| v1_base | V1 | IndoBERT base | — | 0,5210 | 0,2144 | — | 0,5347 | 8,0 mnt | Baseline tanpa cleaning. |
| v1_large | V1 | IndoBERT large | — | 0,7114 | 0,3749 | — | 0,7154 | 25,1 mnt | Large mengungguli base pada V1. |
| v2_base | V2 | IndoBERT base | — | 0,5278 | 0,2120 | — | 0,5190 | 7,3 mnt | Cleaning V2 belum meningkatkan macro F1 base legacy. |
| v2_large | V2 | IndoBERT large | — | 0,7244 | 0,3365 | — | 0,7088 | 22,7 mnt | Akurasi legacy tertinggi sebelum protokol final. |
| v3_base | V3 | IndoBERT base | 10 | 0,8269 | 0,5774 | **0,7898** | 0,8258 | 14,9 mnt | Reliable F1 tertinggi pada split V3. |
| v3_large | V3 | IndoBERT large | 8 | 0,8333 | 0,5917 | 0,7762 | 0,8290 | 38,0 mnt | Macro F1 12 kelas lebih tinggi dari V3 base; reliable F1 sedikit lebih rendah. |
| v4_base | V4 | IndoBERT base | 12 | **0,8487** | 0,5855 | 0,7861 | **0,8463** | 16,9 mnt | Akurasi/weighted F1 tertinggi pada protokol V4. |
| v4_large | V4 | IndoBERT large | 10 | 0,8418 | **0,5944** | 0,7894 | 0,8397 | 44,9 mnt | Macro F1 12 kelas tertinggi; model final untuk cakupan kelas. |

Perbandingan V3 dan V4 perlu dibaca per-metrik, bukan dengan satu angka saja.
`v4_large` berada 0,0027 di bawah `v3_base` pada macro F1 reliable
(0,7894 vs 0,7898), tetapi unggul 0,0170 pada macro F1 12 kelas
(0,5944 vs 0,5774). `v4_base` mencapai akurasi 0,8487, 0,0154 di atas
`v3_large`, sambil menggunakan 1.012,2 detik, bukan 2.282,7 detik. Namun V4
memakai 727 observasi test dibanding 468 pada V3 serta grup pasien/template
yang lebih ketat; angka lintas versi tidak boleh diperlakukan sebagai uji
berpasangan atau klaim kenaikan kausal akibat satu perubahan pipeline.

## Perbandingan baseline EDA V3–V4

Baseline CPU memakai TF-IDF word 1–2 gram dan LinearSVC dengan class balance.
Baseline ini bukan kandidat produksi; fungsinya mendeteksi regresi data sebelum
menghabiskan kuota GPU.

| Dataset | Akurasi | Macro F1 12 kelas | Macro F1 reliable 7 | Makna |
|---|---:|---:|---:|---|
| V3 | 0,8077 | 0,5686 | 0,7604 | Split lama dengan template overlap dan kehilangan baris pasca-split. |
| V4 | 0,8377 | 0,5539 | 0,8067 | Kualitas/split lebih ketat; reliable F1 naik 0,0463. |

Kenaikan baseline reliable F1 setelah gate V4 mendukung keputusan engineering
bahwa noise yang jelas layak ditangani. Turunnya macro F1 semua kelas dari
0,5686 menjadi 0,5539 juga menunjukkan bahwa problem kelas langka belum
selesai. Karena split berubah, baseline ini dipakai sebagai uji regresi
pipeline, bukan sebagai ukuran efek kausal cleaning.

## Performa per kelas dan pola kesalahan

Per-class report final yang tersedia untuk V3 memperlihatkan batas utama
model: model large cukup kuat pada empat kelas mayoritas—COVID-19 (F1 0,7452),
Pneumonia/ISPA (0,7811), Suspek Dengue (0,7381), dan Diare Akut (0,8058)—tetapi
masih rendah pada Acute Flaccid Paralysis (0,4348), Sindrom Jaundice Akut
(0,2000), dan Suspek HFMD (0,3333). Kelas dengan test support 0–4 memperoleh
F1 nol pada run tersebut. Pola ini selaras dengan EDA V4: data dominan
memiliki bukti tekstual lebih banyak, sedangkan kelas kecil memerlukan data
baru, bukan sekadar tuning hyperparameter.

V4 menyimpan `error_analysis.json` per training (confusion matrix, pasangan
kelas terkeliru, dan error rate agregat menurut sumber, jenis kunjungan,
kecocokan anchor, serta panjang teks) dan `run_manifest.json` (hash split,
konfigurasi, lingkungan, checkpoint, dan epoch). Kedua artefak tidak memuat
teks anamnesis atau ID pasien. Artefak tersebut merupakan dasar audit yang
lebih tepat untuk menulis analisis per-kelas V4 setelah hasil model diunduh
dari Hugging Face/Kaggle Output.

## Validitas, keterbatasan, dan langkah berikutnya

Hasil ini menunjukkan validasi internal, bukan generalisasi lintas rumah
sakit. Konsentrasi sumber hampir 100% pada beberapa kelas dan gaya penulisan
rumah sakit dapat membocorkan sinyal yang tidak tersedia di lokasi lain.
Pemecahan grup V4 mencegah kebocoran pasien/template di dalam dataset, tetapi
tidak menggantikan external validation. Label juga merupakan rekonstruksi
mapping ICD, sehingga pemetaan yang ambigu perlu disahkan oleh klinisi/dosen
sebelum model diposisikan sebagai alat pendukung keputusan.

Prioritas eksperimen berikutnya:

1. Tambahkan kasus berlabel dari rumah sakit atau periode waktu lain,
   terutama lima kelas dengan kurang dari 30 observasi.
2. Bekukan V4 dan lakukan test lintas-sumber atau temporal holdout; jangan
   memakai test set V4 untuk tuning berulang.
3. Audit error V4 large pada pasangan COVID–Pneumonia/ISPA–Dengue–Diare;
   perbaiki skema dokumentasi/label bila bukti klinis pembeda memang tidak
   muncul dalam anamnesis.
4. Re-run model hanya setelah perubahan data/protokol dicatat dalam manifest;
   bandingkan hash split dan checkpoint agar hasil dapat direproduksi.

## Reproduksibilitas dan lokasi artefak

- Konfigurasi: `config/experiment.yaml`
- Ringkasan build data: `data/processed/v*/build_summary.json` (private)
- Log metrik: `experiments/runs.jsonl`
- EDA V4 agregat: `reports/v4_engineering_eda.md` dan `reports/v4_eda_metrics.json`
- Report per kelas legacy: `reports/per_kelas_v*_*.md`
- Model private: `gnafhan/pkt-ta-indobert-v3-base`,
  `gnafhan/pkt-ta-indobert-v3-large`, `gnafhan/pkt-ta-indobert-v4-base`, dan
  `gnafhan/pkt-ta-indobert-v4-large` di Hugging Face.

Laporan ini tidak menyalin data pasien maupun credential. Semua angka model
berasal dari run penuh terakhir yang tercatat hingga 12 Agustus 2026.
