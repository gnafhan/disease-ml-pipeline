# Audit Error V4 dan Rancangan Eksperimen V5

## Kesimpulan utama

V4 sudah menyelesaikan masalah integritas split, tetapi belum dapat menyelesaikan keterbatasan informasi klinis dan ketimpangan data. Kedua model V4 mencapai macro F1 12 kelas sekitar 0,59 (`v4_base` 0,5855; `v4_large` 0,5944), sementara macro F1 tujuh kelas reliable sekitar 0,79. Selisih hampir 0,20 ini bukan sekadar masalah optimisasi model: lima kelas dengan jumlah training 6–19 memiliki test support 1–4, sehingga sebuah prediksi dapat mengubah F1 kelas secara drastis.

Kami tidak merekomendasikan V5 sebagai “V4 dengan lebih banyak epoch” atau pembersihan tambahan yang mengejar metrik test. V5 perlu menjadi eksperimen terkontrol: mempertahankan split V4 sebagai benchmark beku, menambahkan audit label dan data baru secara terpisah, menguji ablation fitur manual, dan menambahkan evaluasi selective prediction/abstention.

## Bukti dari audit V4

### 1. Integritas split sudah baik

V4 menulis 4.840 dari 4.840 baris yang masuk ke split. Tidak ada group pasien yang muncul di lebih dari satu subset, tidak ada template teks kanonis yang melintasi subset, dan tidak ada grup template dengan label konflik pada data akhir. Masih ada 32 grup template berulang berisi 84 baris, tetapi setiap grup sepenuhnya berada pada satu subset dan memiliki satu label. Memperbaiki deduplikasi atau menambah leak guard bukan tuas utama untuk peningkatan V5.

| Invarian | V3 | V4 | Keputusan V5 |
|---|---:|---:|---|
| Baris hilang saat split | 584 | 0 | Pertahankan grouped split V4. |
| Patient-overlap group | 0 | 0 | Aturan pasien tidak diubah. |
| Template-overlap group | 12 | 0 | Jangan kembali ke split per baris. |
| Grup template berlabel konflik | 2 | 0 | Konflik eksplisit sudah ditangani. |
| Catatan anamnesis kosong | 87 | 0 | Cleaning kosong sudah efektif. |

### 2. Ketimpangan kelas membatasi macro F1 seluruh 12 kelas

Distribusi training V4 membentang dari 1.103 contoh Pneumonia/ISPA hingga 6 contoh Suspek Meningitis/Ensefalitis. Loss saat ini memakai bobot `1/sqrt(n+1)`: relatif terhadap Pneumonia/ISPA, bobot Meningitis hanya 12,56 kali lebih besar, sementara rasio jumlah contoh mencapai 183,8 kali. Focal loss membantu memberi perhatian pada contoh sulit, tetapi tidak menciptakan variasi kasus untuk kelas dengan enam contoh.

| Kelas | Train | Test | Status interpretasi |
|---|---:|---:|---|
| Pneumonia/ISPA | 1.103 | 236 | Stabil secara relatif. |
| COVID-19 Konfirmasi | 740 | 159 | Stabil secara relatif, tetapi satu sumber. |
| Diare Akut | 564 | 121 | Stabil secara relatif. |
| Suspek Dengue | 536 | 115 | Stabil secara relatif. |
| Acute Flaccid Paralysis | 232 | 50 | Dapat dievaluasi, tetapi heterogen. |
| Sindrom Jaundice Akut | 96 | 21 | Interval ketidakpastian masih lebar. |
| Suspek HFMD | 52 | 12 | Masih terbatas. |
| Suspek Tetanus | 19 | 4 | Insufficient-data. |
| GHPR | 17 | 4 | Insufficient-data. |
| Diare Berdarah | 10 | 2 | Insufficient-data. |
| Suspek Leptospirosis | 10 | 2 | Insufficient-data. |
| Suspek Meningitis/Ensefalitis | 6 | 1 | Insufficient-data. |

V5 tidak boleh mengklaim F1 kelas langka sebagai bukti kesiapan klinis. Model tetap dapat memuat 12 kelas untuk riset, tetapi laporan harus menampilkan support dan interval bootstrap per kelas; prediksi berkeyakinan rendah pada kelas berisiko dialihkan ke review manusia.

### 3. Bukti teks tidak selalu cocok dengan label sindrom

Sebanyak 536 dari 4.840 baris (11,07%) tidak memuat anchor gejala yang selaras dengan kelasnya. Anchor bukan ground truth dan tidak boleh digunakan untuk auto-relabel, tetapi angka ini menandai catatan yang patut diperiksa klinis. Masalah paling terkonsentrasi terdapat pada Sindrom Jaundice Akut (62/138; 44,9%) dan Suspek Meningitis/Ensefalitis (5/8; 62,5%). Pada test V4, 12 dari 21 catatan Jaundice (57,1%) dan satu-satunya catatan Meningitis tidak memiliki anchor.

| Kelas | Anchor tidak cocok | Persentase | Risiko yang perlu diuji |
|---|---:|---:|---|
| Pneumonia/ISPA | 188/1.575 | 11,9% | Catatan pendek atau istilah respirasi non-standar. |
| COVID-19 Konfirmasi | 91/1.058 | 8,6% | Label konfirmasi dapat hadir tanpa gejala spesifik. |
| Acute Flaccid Paralysis | 50/332 | 15,1% | Variasi narasi neurologis. |
| Sindrom Jaundice Akut | 62/138 | 44,9% | Bukti relevan sering tidak tertulis dalam anamnesis. |
| Suspek Meningitis/Ensefalitis | 5/8 | 62,5% | Data sangat kecil dan teks tidak cukup. |

Sebanyak 27 Pneumonia/ISPA masih memiliki dua token atau kurang setelah V4, dibanding tiga Diare Akut dan dua COVID-19. V5 perlu menyimpan bin panjang teks dalam laporan per kelas; model teks tidak dapat diharapkan membedakan diagnosis kompleks ketika masukan hanya dua token.

### 4. Source shift tetap merupakan ancaman validitas

Sembilan dari 12 kelas memiliki lebih dari 68% contoh dari satu sumber; COVID-19 Konfirmasi, GHPR, dan Suspek Meningitis/Ensefalitis seluruhnya berasal dari satu sumber. Kolom `source` tidak diberikan ke model, tetapi gaya dokumentasi sumber masih hidup di anamnesis. Baseline source/visit type tanpa teks memperoleh akurasi 0,2242 pada V4, jauh di atas tebakan seragam untuk 12 kelas (0,0833). Angka ini tidak membuktikan model utama memakai sumber, tetapi cukup kuat untuk menuntut evaluasi source-holdout atau temporal holdout sebelum klaim generalisasi.

| Kelas | Sumber dominan | Pangsa sumber dominan |
|---|---|---:|
| COVID-19 Konfirmasi | RS Akademik UGM | 100,0% |
| Pneumonia/ISPA | RSUD NAS | 97,4% |
| Diare Akut | RSUD NAS | 99,1% |
| Sindrom Jaundice Akut | RS Akademik UGM | 85,5% |
| Acute Flaccid Paralysis | RSUD NAS | 72,3% |
| Suspek Dengue | RS Akademik UGM | 52,7% |

### 5. Fitur manual saat ini tidak simetris antar kelas

Input model menyisipkan token buatan manusia untuk Dengue, COVID, dan tanda respirasi berat sebelum anamnesis mentah. Cakupannya tidak merata: flag Dengue ada pada 51,6% baris Suspek Dengue, flag COVID pada 40,1% baris COVID-19, sementara sembilan kelas lain tidak memiliki keluarga flag khusus. Sebagian flag juga muncul lintas kelas—flag COVID pada 9,5% Pneumonia/ISPA dan 8,2% Diare Akut. Fitur ini tidak terbukti salah, tetapi merupakan hipotesis yang belum diuji melalui ablation. V5 harus membandingkan raw-anamnesis+metadata dengan raw-anamnesis+metadata+flag pada split beku yang sama.

### 6. Audit prediksi V4 belum portabel

Pipeline membuat `error_analysis.json` dan `run_manifest.json` setelah setiap training. Artefak pertama berisi confusion matrix dan error slice menurut sumber, jenis kunjungan, anchor, serta panjang teks; keduanya ikut terunggah ke repo Hugging Face karena pola upload mencakup `*.json`. Namun kernel Kaggle hanya mengekspor `runs.jsonl` dan matriks ringkas. GitHub juga hanya menyimpan log metrik. Akibatnya, workspace ini tidak memuat confusion matrix aktual `v4_base`/`v4_large`; audit ini dapat mengidentifikasi penyebab struktural, tetapi belum dapat mengurutkan pasangan salah-klasifikasi model V4 yang sebenarnya.

V5 harus mengekspor artefak agregat tersebut ke Kaggle Output dan, bila tidak mengandung teks/ID pasien, commit ke GitHub. Ini adalah perbaikan audit, bukan perubahan model.

## Desain V5 yang direkomendasikan

### Prinsip

1. **Bekukan benchmark V4.** Train/validation/test V4, seed, dan grouped split diperlakukan sebagai benchmark internal tetap. Tidak ada keputusan cleaning, relabel, atau hyperparameter yang memakai hasil test V4.
2. **Pisahkan perbaikan data dari optimisasi model.** Semua review label terjadi pada training set atau pada batch data baru; test V4 tetap tak tersentuh.
3. **Jangan menyamarkan kelas langka.** Model masih mengeluarkan 12 kelas, tetapi sistem juga mengeluarkan confidence dan opsi abstain/review untuk kelas atau prediksi berisiko tinggi.
4. **Gunakan gate murah sebelum GPU.** Baseline CPU, test integritas split, dan audit label harus lulus sebelum kuota GPU dipakai.

### Matriks eksperimen V5

| Tahap | Perubahan tunggal | Tujuan | Kriteria lanjut |
|---|---|---|---|
| V5-A: audit artefak | Ekspor `error_analysis.json`, `run_manifest.json`, prediksi agregat aman, dan confidence bin dari V4. | Mengukur confusion nyata dan kalibrasi tanpa retraining. | Semua artefak tersedia; tidak ada teks/ID pasien. |
| V5-B: ablation input | Bandingkan input raw anamnesis+usia+sex+bulan **tanpa flag manual** dengan konfigurasi V4 ber-flag pada split V4 beku. | Menguji apakah flag asimetris membantu atau mengunci shortcut. | Pilih input dari validation reliable F1 dan slice error, bukan test. |
| V5-C: loss ablation | Bandingkan focal+inverse-square-root (V4) dengan class-balanced loss atau logit-adjustment; hyperparameter ditentukan di validation. | Memeriksa recall kelas minoritas tanpa merusak tujuh kelas reliable. | Laporkan macro all, macro reliable, dan recall per kelas beserta support. |
| V5-D: selective prediction | Kalibrasi threshold pada validation dan keluarkan `review_required` jika confidence di bawah threshold. | Mengurangi prediksi salah berkeyakinan rendah. | Laporkan coverage, selective accuracy, dan error rate kasus yang diterima. |
| V5-E: data adjudikasi | Tinjau klinisi seluruh kelas <30, seluruh catatan anchor-miss Jaundice/Meningitis, serta sampel acak kelas mayoritas; simpan keputusan dan alasan. | Memisahkan label ambiguity dari keterbatasan model. | Review tidak menyentuh test V4; perubahan label berversi. |
| V5-F: generalisasi | Tambahkan batch periode terbaru atau sumber rumah sakit lain, lalu lakukan temporal/source holdout. | Mengukur external validity. | Tidak mengklaim siap lintas-rumah-sakit sebelum evaluasi ini ada. |

Urutan ini memprioritaskan V5-A dan V5-B. V5-A tidak memerlukan GPU; V5-B dapat menjawab apakah fitur buatan manusia benar-benar membantu dengan ablation terkontrol. V5-C baru layak dijalankan jika baseline V5-B stabil. V5-E dan V5-F adalah sumber perbaikan terbesar, tetapi membutuhkan otoritas klinis dan data baru; keduanya tidak boleh disimulasikan dengan duplikasi atau oversampling pada validation/test.

## Definisi keberhasilan V5

| Dimensi | Syarat |
|---|---|
| Integritas | 0 patient overlap, 0 template overlap, 0 baris hilang saat split. |
| Performa utama | Macro F1 reliable tidak turun melebihi margin yang ditetapkan sebelum run; macro F1 12 kelas dan weighted F1 dilaporkan bersama. |
| Kelas minoritas | Recall dan support per kelas ditampilkan; tidak ada klaim stabil untuk test support <10. |
| Kalibrasi | Coverage dan error rate setelah abstention tersedia. |
| Robustness | Error slice menurut sumber, visit type, anchor, dan panjang teks tersedia. |
| Reproduksibilitas | Hash split/config, manifest runtime, dan artefak error agregat ikut diekspor. |

## Hal yang tidak direkomendasikan

- Menghapus lima kelas langka agar macro F1 tampak naik.
- Mengubah label menggunakan keyword atau hasil prediksi model.
- Menggunakan test V4 berulang kali untuk memilih threshold, loss, atau preprocessing.
- Menggandakan baris ke validation/test sebagai penyeimbang.
- Menambahkan `source` sebagai fitur prediktif untuk mengejar akurasi internal.
- Menyebut hasil sebagai diagnosis klinis atau validasi eksternal.

## Keputusan praktis

V5 paling kuat sebagai **V4 benchmark beku + audit prediksi lengkap + ablation input terkontrol + selective prediction**, lalu diikuti data adjudikasi. Jika harus memilih satu pekerjaan sebelum menggunakan GPU lagi, lakukan V5-A: ambil dan ekspor `error_analysis.json` dari dua repo model V4. Itu akan menjawab apakah error `v4_large` terutama terkonsentrasi pada COVID–Pneumonia, Dengue–Diare, kelas pendek, sumber tertentu, atau catatan tanpa anchor. Tanpa artefak itu, perubahan loss/model akan menjadi tebakan.
