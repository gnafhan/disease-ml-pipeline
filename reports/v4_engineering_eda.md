# Laporan Engineering EDA & Keputusan Data V4

Di-generate oleh `python -m src.eda_v4`. Report ini hanya memuat statistik agregat; teks anamnesis dan identifier pasien tidak pernah ditulis ke sini.

## Hasil

V4 mempertahankan taksonomi 12 kelas, tetapi mengubah quality gate dan cara split. Pipeline menulis 4,840 baris (3,385/728/727) dan tidak kehilangan baris saat split. V3 menghasilkan 5.249 baris bersih tetapi hanya menulis 4,665; 584 baris (11.1%) dibuang oleh leak guard setelah split.

Baseline teks CPU naik dari macro F1 kelas reliable 0.7604 menjadi 0.8067. Ini engineering gate, bukan perbandingan model final: V4 memakai test split patient/template-grouped yang lebih ketat dan test set yang lebih besar.

## Failure mode yang ditemukan di V3

- **Baris hilang setelah split:** 584. Algoritma lama membagi baris lebih dulu, lalu menghapus kunjungan val/test bila pasiennya muncul di subset sebelumnya. V4 menetapkan connected group pasien/template sebelum split.
- **Header Excel bergeser:** delapan sheet RSUD menaruh header asli pada baris Excel 2–7. Loader lama melewatkan 25 baris pasien. Deteksi header adaptif menaikkan raw ingestion dari 5.908 menjadi 5,933 baris.
- **Anamnesis kosong:** 87 baris tersimpan tanpa sinyal teks.
- **Duplikasi template:** 449 baris masuk ke 166 template ternormalisasi yang berulang.
- **Teks identik dengan label konflik:** 91 baris dalam 2 group template mempunyai lebih dari satu label. Mayoritas berupa catatan kosong; satu template kontrol post-ranap generik juga melintasi label.
- **Pasien multi-kondisi:** 252 baris dari 87 ID pasien mencakup lebih dari satu kelas. Baris ini tidak otomatis salah, tetapi wajib berada dalam subset yang sama.
- **Konsentrasi sumber:** beberapa kelas hampir seluruhnya berasal dari satu rumah sakit. Random internal split mengukur in-domain recognition, bukan generalisasi lintas rumah sakit.
- **Evaluasi kelas langka:** lima kelas V4 tetap mempunyai kurang dari 30 baris. Test support-nya hanya 1–4 kasus per kelas, sehingga satu prediksi bisa mengubah F1 secara ekstrem.

## Tindakan quality gate V4

| action | rows |
| --- | --- |
| conflicting_template_labels | 6 |
| control_without_clinical_signal | 1 |
| duplicate_patient_visit | 310 |
| empty_text | 106 |
| kept | 4840 |
| short_text_without_clinical_signal | 9 |

Total baris yang dibuang aturan khusus V4: 432. Gate catatan kontrol/sangat pendek memakai sinyal klinis label-agnostic. Kecocokan anchor terhadap kelas hanya dihitung sebagai diagnostik; nilainya tidak pernah menentukan kept/drop atau mengubah label.

## Profil kelas, sumber, dan sinyal teks

| kelas | baris V3 | baris V4 | sumber terbesar V4 | cakupan anchor kelas | >=30 baris |
| --- | --- | --- | --- | --- | --- |
| Pneumonia/ISPA | 1375 | 1575 | 97.4% RSUD_NAS | 88.1% | yes |
| COVID-19 Konfirmasi | 1054 | 1058 | 100.0% RS_AKADEMIK_UGM | 91.4% | yes |
| Suspek Dengue | 858 | 766 | 52.7% RS_AKADEMIK_UGM | 92.8% | yes |
| Diare Akut | 750 | 806 | 99.1% RSUD_NAS | 92.6% | yes |
| Acute Flaccid Paralysis | 351 | 332 | 72.3% RSUD_NAS | 84.9% | yes |
| Sindrom Jaundice Akut | 132 | 138 | 85.5% RS_AKADEMIK_UGM | 55.1% | yes |
| Suspek HFMD | 65 | 76 | 68.4% RSUD_NAS | 82.9% | yes |
| GHPR | 25 | 25 | 100.0% RSUD_NAS | 96.0% | no |
| Suspek Tetanus | 24 | 28 | 85.7% RS_AKADEMIK_UGM | 78.6% | no |
| Suspek Leptospirosis | 14 | 14 | 92.9% RS_AKADEMIK_UGM | 78.6% | no |
| Diare Berdarah | 9 | 14 | 92.9% RSUD_NAS | 85.7% | no |
| Suspek Meningitis/Ensefalitis | 8 | 8 | 100.0% RSUD_NAS | 37.5% | no |

Source share mendekati 100% adalah deployment risk: kelas dan gaya dokumentasi saling terikat. V4 mempertahankan baris tersebut, tetapi melakukan stratifikasi `class x source` sambil mengelompokkan pasien/template. Pengukuran performa eksternal tetap membutuhkan rumah sakit lain atau periode waktu yang lebih baru.

## Baseline CPU murah

Baseline memakai TF-IDF word 1–2 gram dan class-balanced LinearSVC. Ini bukan model produksi; fungsinya menangkap regresi pipeline sebelum memakai kuota GPU.

| dataset | accuracy | macro F1 semua 12 | macro F1 reliable 7 |
| --- | --- | --- | --- |
| V3 | 0.8077 | 0.5686 | 0.7604 |
| V4 | 0.8377 | 0.5539 | 0.8067 |

Arah confusion terbesar V4:

| kelas aktual | kelas prediksi | baris |
| --- | --- | --- |
| Pneumonia/ISPA | Suspek Dengue | 12 |
| Diare Akut | Pneumonia/ISPA | 11 |
| Suspek Dengue | COVID-19 Konfirmasi | 11 |
| COVID-19 Konfirmasi | Pneumonia/ISPA | 10 |
| COVID-19 Konfirmasi | Suspek Dengue | 8 |
| Suspek Dengue | Diare Akut | 8 |
| Suspek Dengue | Pneumonia/ISPA | 6 |
| Acute Flaccid Paralysis | COVID-19 Konfirmasi | 5 |
| COVID-19 Konfirmasi | Diare Akut | 5 |
| Pneumonia/ISPA | Diare Akut | 5 |

Overlap dominan tetap berada pada COVID-19, Pneumonia/ISPA, Diare Akut, dan Suspek Dengue. Catatan mereka sama-sama memuat demam, batuk, mual/muntah, dan diare. Cleaning dapat membuang noise yang jelas, tetapi tidak dapat menciptakan bukti klinis diskriminatif yang memang tidak pernah dicatat.

## Diagnostik metadata/sumber

Usia/sex/bulan saja menghasilkan reliable macro F1 0.0937 pada V3 dan 0.0955 pada V4. Source/visit type saja menghasilkan 0.1079 dan 0.1061. Accuracy source-only V3 sebesar 0.3355, terutama karena COVID-19 hanya ada di RS Akademik UGM; pada split V4 turun menjadi 0.2242. Kolom `source` sengaja tidak diberikan ke neural model, tetapi gaya teks spesifik rumah sakit masih dapat membuka shortcut yang sama.

## Invariant V4

| pemeriksaan | V3 | target/hasil V4 |
| --- | --- | --- |
| rows lost during split | 584 | 0 |
| patient overlap groups | 0 | 0 |
| template overlap groups | 12 | 0 |
| empty anamnesis | 87 | 0 |
| conflicting template groups | 2 | 0 |
| class count | 12 | 12 |

## Keputusan yang sengaja tidak diambil

- V4 tidak menghapus kelas langka untuk menaikkan headline metric.
- V4 tidak me-relabel kelas berbasis ICD menggunakan keyword.
- V4 tidak melakukan oversampling dengan menyalin baris ke validation/test.
- V4 tidak mengklaim validasi lintas rumah sakit dari random internal split.
- V4 tidak memasukkan contoh teks mentah ke report karena sumber berisi data pasien.
- V4 menulis `record_id` sebagai SHA-256 pseudonym, bukan nomor RM sumber; dataset tetap diperlakukan sensitif dan private.

## Artefak audit setiap training

Selain `runs.jsonl` dan model, setiap run sekarang menulis `error_analysis.json` berisi confusion matrix, confusion pair terbesar, serta error rate agregat menurut source, visit type, kecocokan anchor, dan panjang teks. `run_manifest.json` merekam SHA-256 train/val/test dan config, git SHA, versi Python/library, device/GPU, checkpoint terbaik, global step, dan epoch selesai. Keduanya tidak menyimpan teks anamnesis atau ID pasien.

Dependency di `requirements.txt` belum dikunci sampai patch version, sehingga manifest runtime wajib disimpan bersama hasil. Untuk klaim final, bandingkan hash manifest dan pastikan load-best-checkpoint menghasilkan metrik yang sama ketika model di-reload.

## Gate eksperimen GPU

1. Jalankan smoke-test `v4_base` saja.
2. Jalankan full `v4_base`; persist `runs.jsonl`, metrik per kelas, confusion matrix, dan model sebelum kernel berhenti.
3. Bandingkan recall per kelas dan reliable-class macro F1 dengan V3, sambil mencatat bahwa split berubah.
4. Jalankan `v4_large` hanya jika V4 base stabil dan expected gain layak untuk sekitar 40–60 menit GPU.
5. Perlakukan lima kelas di bawah 30 baris sebagai insufficient-data classes terlepas dari test F1 yang volatil.
