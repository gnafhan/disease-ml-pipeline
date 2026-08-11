"""
Robustness & reproducibility check dasar. Jalankan: pytest tests/

Ini bukan pelengkap -- test_row_count_matches_source() persis nge-tes
supaya bug sheet-loading lama (cuma sheet pertama ter-load, ~60% data
hilang tanpa disadari) tidak bisa terulang tanpa ketahuan.
"""


def test_row_count_matches_source():
    """TODO: ambil satu file RSUD NAS (misal ISPA.xlsx yang punya banyak
    sheet), hitung total baris SEMUA sheet manual pakai openpyxl langsung,
    bandingkan dengan hasil src.ingest.load_rsud_nas_folder().
    assert sama persis -- kalau beda, pipeline harus gagal jelas.
    """
    pass


def test_pipeline_reproducibility():
    """TODO: jalankan ingest+label+clean 2x dengan input identik,
    assert row count & distribusi kelas hasil akhir identik.
    """
    pass


def test_malformed_file_handled():
    """TODO: buat 1 file Excel dummy dengan kolom hilang / sheet kosong,
    assert pipeline raise error yang JELAS (bukan diam-diam skip data
    seperti bug lama).
    """
    pass
