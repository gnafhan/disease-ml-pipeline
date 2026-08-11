"""
Ingest & Validate -- Tahap 02 pipeline.

Baca SEMUA sheet dari tiap file Excel RSUD NAS (Rawat Jalan + Rawat Inap)
dan file RS Akademik UGM. Validasi jumlah baris ter-load vs jumlah baris
di file sumber (assert, jangan cuma print) -- ini yang dulu jadi bug besar
(sheet-loading, cuma sheet pertama ter-load, ~60% data hilang tanpa disadari).

Port logika dari: v6_training.py (bagian load dataset), STORY_DEVELOPMENT.md
bagian "Dataset V4 -- Re-merge dengan ICD".

PERHATIAN -- inkonsistensi nama file yang ditemukan saat audit raw_data/:
  - Rawat Jalan pakai "Suspek HFMD.xlsx", Rawat Inap pakai "Suspek HMFD.xlsx"
  - Rawat Inap pakai "Ranap Campak.xlsx", Rawat Jalan pakai "Rajal Campak.xlsx" / "Suspek Campak.xlsx"
  - Rawat Inap TIDAK punya semua kategori yang ada di Rawat Jalan (GHPR, ILI,
    Jaundice, Leptospirosis, Meningitis/Ensefalitis cuma ada di Rawat Jalan)
Jangan exact-match nama file -- normalize dulu (uppercase, strip spasi) dan
JANGAN asumsikan Rawat Jalan & Rawat Inap punya set kategori yang sama.
"""
from pathlib import Path
import pandas as pd


def load_rsud_nas_folder(folder: Path) -> pd.DataFrame:
    """Baca SEMUA file .xlsx di folder (Rawat Jalan ATAU Rawat Inap),
    baca SEMUA sheet per file. Tiap sheet = satu kode ICD-10 (nama sheet).

    Return DataFrame gabungan dengan kolom minimal:
        source_file, sheet_name (icd_code), <kolom asli tiap sheet>

    TODO:
      1. iterate semua .xlsx di `folder`
      2. pd.ExcelFile(path).sheet_names -- iterate SEMUA, bukan cuma sheet [0]
      3. tag tiap baris dengan source_file & sheet_name
      4. panggil validate_row_count() sebelum return
    """
    raise NotImplementedError


def load_rsa_ugm(filepath: Path) -> pd.DataFrame:
    """Baca 'Penelitian Wawa 1 Labeled.xlsx'.
    Kolom penting: Subjective (teks anamnesa SOAP), ICD 10, Diagnosis, sindrom.
    TODO: port dari v6_training.py, bersihkan header KU/RPS/RPD dari teks SOAP.
    """
    raise NotImplementedError


def validate_row_count(df: pd.DataFrame, source_path: Path, expected_sheet_count: int) -> None:
    """Robustness check -- panggil di akhir SETIAP load_* function.
    RAISE (bukan warning) kalau jumlah sheet/baris ter-load meleset dari sumber.
    Ini yang jadi test_ingest.py::test_row_count_matches_source.
    """
    raise NotImplementedError
