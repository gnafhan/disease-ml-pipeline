"""
Test ini jalan terhadap DATA ASLI (raw_data/) -- makanya cuma bisa dijalankan
di komputer yang punya folder raw_data (bukan di CI/cloud generik). Kalau
raw_data tidak ada, semua test di file ini di-skip otomatis (bukan gagal).

Jalankan: pytest tests/test_ingest.py -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ingest, label as label_mod
from src.split import load_config

# Test dijalankan dari pipeline/, sedangkan raw_data sengaja berada satu level
# di atas agar tidak tercampur repo/push GitHub.
RAW_DATA_AVAILABLE = os.path.isdir("../raw_data")

pytestmark = pytest.mark.skipif(not RAW_DATA_AVAILABLE, reason="raw_data/ tidak ada di environment ini")


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/experiment.yaml")


def test_row_count_matches_source(cfg):
    """
    Cek self-consistency: total baris yang di-load harus sama dengan total baris
    kalau dihitung manual langsung dari file .xlsx (independen dari kode ingest.py).
    Ini yang akan menangkap bug seperti V2 (cuma sheet pertama ter-load).
    """
    folder = cfg["paths"]["raw_rsud_nas_rawat_jalan"]
    manual_total = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".xlsx"):
            continue
        xl = pd.ExcelFile(os.path.join(folder, fname))
        for sheet in xl.sheet_names:
            grid = xl.parse(sheet, header=None)
            header_row = 0
            for idx, row in grid.head(15).iterrows():
                values = {str(value).strip() for value in row.dropna()}
                if "Anamnesa" in values and values & {"Kode ICD", "ICD Code", "No. RM"}:
                    header_row = int(idx)
                    break
            manual_total += len(xl.parse(sheet, header=header_row))

    result = ingest.load_rsud_nas_folder(folder, "RAWAT JALAN")
    assert len(result.df) == manual_total, (
        f"ingest.py cuma load {len(result.df)} baris, seharusnya {manual_total} "
        "(dihitung manual dari semua sheet) -- ada sheet yang ke-skip."
    )


def test_rs_akademik_row_count(cfg):
    result = ingest.load_rsa_ugm(cfg["paths"]["raw_rsa_ugm"])
    assert len(result.df) == 1857, f"Expected 1857 baris RS Akademik UGM, got {len(result.df)}"


def test_pipeline_reproducibility(cfg):
    """Load 2x dengan seed sama -> hasil harus identik (row count & distribusi kelas)."""
    df1 = ingest.load_all_raw(cfg["paths"])
    df2 = ingest.load_all_raw(cfg["paths"])
    assert len(df1) == len(df2)

    labeled1, _ = label_mod.apply_labels(df1)
    labeled2, _ = label_mod.apply_labels(df2)
    assert labeled1["final_class"].value_counts().equals(labeled2["final_class"].value_counts())


def test_labeled_classes_within_locked_taxonomy(cfg):
    """Semua final_class yang dihasilkan HARUS ada di daftar 12 kelas config -- no silent new class."""
    raw_df = ingest.load_all_raw(cfg["paths"])
    labeled_df, _ = label_mod.apply_labels(raw_df)
    unexpected = set(labeled_df["final_class"].unique()) - set(cfg["classes"])
    assert not unexpected, f"Kelas di luar taksonomi terkunci: {unexpected}"


def test_malformed_file_handled(tmp_path):
    """Folder kosong/tidak ada -> harus raise error yang jelas, bukan diam-diam return kosong."""
    empty_folder = tmp_path / "RAWAT JALAN KOSONG"
    empty_folder.mkdir()
    with pytest.raises(RuntimeError):
        ingest.load_rsud_nas_folder(str(empty_folder), "RAWAT JALAN")

    with pytest.raises(FileNotFoundError):
        ingest.load_rsud_nas_folder(str(tmp_path / "folder_tidak_ada"), "RAWAT JALAN")
