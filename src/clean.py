"""
Clean -- Tahap 04. Aturan pembersihan noise. Parametrized lewat argumen,
JANGAN hardcode angka ambang di sini kalau bisa ditaruh di config.

Aturan WAJIB (sudah tervalidasi di STORY_DEVELOPMENT.md):

1. Hapus baris kontrol/pasca-rawat TANPA keluhan akut aktif.
   Baris kontrol yang MASIH ada keluhan aktif (misal "kontrol, batuk (+)")
   TETAP DIPERTAHANKAN -- itu masih informatif.

2. Hapus kasus COVID insidental: pasien datang untuk obstetri/bedah,
   kebetulan antigen positif saat skrining masuk RS -- anamnesa-nya BUKAN
   tentang gejala COVID.

3. Merge Pneumonia -> "Pneumonia/ISPA" (secara anamnesa saja, tanpa foto
   toraks/auskultasi, dua ini tidak bisa dibedakan -- F1 Pneumonia = 0.00
   di semua versi model sebelumnya).

4. Kelas ultra-jarang: JANGAN dihapus diam-diam. flag_unreliable_classes()
   menandai kolom `reliable=False`, baris TETAP ADA di dataset -- supaya
   reporting transparan, bukan angka yang disembunyikan.
"""
import pandas as pd


def remove_control_visits_without_complaint(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: filter baris dengan pola teks kontrol/pasca-rawat DAN tidak ada
    keyword keluhan aktif. Catat & return juga jumlah baris yang dibuang
    (untuk metrik audit 'noise removal rate')."""
    raise NotImplementedError


def remove_incidental_covid(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: filter baris label COVID-19 dengan konteks anamnesa obstetri/bedah
    (bukan gejala pernapasan/COVID)."""
    raise NotImplementedError


def merge_pneumonia_into_ispa(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: rename label 'Pneumonia' -> 'Pneumonia/ISPA'."""
    raise NotImplementedError


def flag_unreliable_classes(df: pd.DataFrame, label_col: str, min_support: int) -> pd.DataFrame:
    """Tambah kolom boolean `reliable` berdasarkan jumlah sampel per kelas
    di TRAINING SET (bukan keseluruhan data). TIDAK menghapus baris apapun.
    """
    raise NotImplementedError
