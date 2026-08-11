"""
Label -- Tahap 03. Mapping ke diagnosis ICD-10, BUKAN sindrom SKDR.

Kenapa ini kritis: sindrom SKDR (misal "Suspek Campak" = demam+ruam) itu
kategori SURVEILANS, bukan diagnosis klinis. 95% baris "Suspek Campak" di
data lama sebenarnya Dengue Fever (ICD A90) -- karena petechiae dengue
dianggap "ruam" secara definisi SKDR.

ATURAN: label SELALU dari kolom ICD-10 / sheet_name (isinya kode ICD),
JANGAN dari nama file atau kolom 'sindrom'.

Port logika dari: PLAN_V6_DATA_FIX.md ("Step 1: Fix Label Suspek Campak").
"""
import pandas as pd


def load_icd_mapping(mapping_path) -> pd.DataFrame:
    """Baca 'ICD SKDR versi 30 Desember 2024 (1).xlsx' -- mapping resmi
    ICD-10 -> kelas final. TODO: parse struktur file mapping.
    """
    raise NotImplementedError


def map_icd_to_class(df: pd.DataFrame, icd_mapping: pd.DataFrame) -> pd.DataFrame:
    """Join df (kolom icd_code / sheet_name) ke icd_mapping, hasilkan kolom
    baru `label_class` (salah satu dari config.classes).
    Baris dengan ICD yang tidak relevan (contoh lama: HIV B20, Herpes B00)
    di-drop di sini, DICATAT jumlahnya (untuk metrik audit "noise removal rate").
    TODO: implementasi join + logging jumlah baris ter-drop.
    """
    raise NotImplementedError
