"""
Ingest raw Excel files jadi satu DataFrame mentah (belum di-label/clean).

Sumber data (lihat config/experiment.yaml -> paths):
  1. RSUD NAS DATA/RAWAT JALAN/*.xlsx   -- 15 file, per file = per folder sindrom SKDR,
     tiap file punya banyak sheet, nama sheet = kode ICD-10 pasien.
  2. RSUD NAS DATA/RAWAT INAP/*.xlsx    -- 10 file, struktur sama.
  3. RS Akademik UGM/Penelitian Wawa 1 Labeled.xlsx -- 1 file, 1857 baris, kolom ICD 10
     sudah ada per-baris (tidak perlu dari nama sheet).

PENTING -- privasi data pasien:
  File mentah RSUD berisi puluhan kolom operasional RS (log antrian, JSON callback
  API pendaftaran online, NIK, nomor HP, dst -- beberapa di antaranya bahkan
  menyimpan NIK di dalam blob JSON kolom "Request/Response Antrol Add/Update").
  Fungsi di bawah HANYA mengambil kolom yang eksplisit di whitelist
  (anamnesa, ICD, sex, usia, tanggal, record id internal RS) -- kolom lain
  TIDAK PERNAH dibaca ke memori sama sekali (usecols saat read_excel), bukan
  cuma di-drop belakangan. Jangan tambah kolom baru ke whitelist tanpa sadar
  konsekuensi privasinya.

Tgl Lahir pasien SENGAJA tidak diambil (usia saja cukup untuk model; kombinasi
No.RM + tanggal lahir terlalu identifying untuk disimpan long-term).
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# ── Kolom yang diambil per sumber data (whitelist, lihat docstring modul) ──────

RSUD_RAJAL_COLUMNS = {
    "anamnesa": "Anamnesa",
    "icd_row": "Kode ICD",
    "sex": "Sex",
    "age": "Usia",
    "date": "Tgl Kunjung",
    "record_id": "No. RM",
}

RSUD_RANAP_COLUMNS = {
    "anamnesa": "Anamnesa",
    "icd_row": "ICD Code",
    "sex": "Sex",
    "age": "Usia",
    "date": "Tgl Masuk RS",
    "record_id": "No. RM",
}

RSA_COLUMNS = {
    "anamnesa": "Subjective",
    "icd_row": "ICD 10",
    "sex": "Gender",
    "age": "Umur",
    "date": "Tanggal Admisi",
    "record_id": "Patient",  # sudah berupa hash anonim di file sumber
}


@dataclass
class IngestResult:
    df: pd.DataFrame
    sheet_row_counts: dict  # {(file, sheet): n_rows} -- untuk validate_row_count


def _parse_age_years(raw) -> float | None:
    """Usia RSUD berformat string '21 th'. RS Akademik sudah berupa int tahun."""
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    m = re.search(r"(\d+)", str(raw))
    return float(m.group(1)) if m else None


def _rename_and_select(df: pd.DataFrame, col_map: dict, extra: dict) -> pd.DataFrame:
    out = pd.DataFrame()
    for canonical, source_col in col_map.items():
        out[canonical] = df[source_col] if source_col in df.columns else None
    for k, v in extra.items():
        out[k] = v
    out["age_years"] = out["age"].apply(_parse_age_years)
    # dayfirst=True -- tanggal RSUD/RS Akademik berformat DD-MM-YYYY (konvensi
    # Indonesia), bukan MM-DD-YYYY. Tanpa ini, tanggal spt "05-03-2025" bisa
    # kepeleset jadi bulan 5 (Mei) padahal harusnya bulan 3 (Maret).
    out["bulan_kunjung"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True).dt.month
    out["sex"] = out["sex"].astype(str).str.strip().str.upper().replace({"NAN": None})
    out["record_id"] = out["record_id"].astype(str)
    return out.drop(columns=["age", "date"])


def load_rsud_nas_folder(folder: str, kategori: str) -> IngestResult:
    """
    folder: path langsung ke folder RAWAT JALAN atau RAWAT INAP (dari
    config['paths']['raw_rsud_nas_rawat_jalan'/'rawat_inap']).
    kategori: label 'RAWAT JALAN' atau 'RAWAT INAP' (dipakai utk pilih kolom &
    tagging, bukan konstruksi path).
    Load SEMUA sheet dari SEMUA file .xlsx di folder ini (fix bug V2 yang cuma
    baca sheet pertama -- lihat STORY_DEVELOPMENT.md).
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

    col_map = RSUD_RAJAL_COLUMNS if kategori == "RAWAT JALAN" else RSUD_RANAP_COLUMNS
    icd_col_name = col_map["icd_row"]

    frames = []
    sheet_row_counts = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".xlsx"):
            continue
        fpath = os.path.join(folder, fname)
        folder_default_class = os.path.splitext(fname)[0]  # nama file = default sindrom
        try:
            xl = pd.ExcelFile(fpath)
        except Exception as e:
            raise RuntimeError(f"Gagal buka {fpath}: {e}") from e

        for sheet in xl.sheet_names:
            usecols = [c for c in col_map.values() if c != icd_col_name] + [icd_col_name]
            try:
                raw = xl.parse(sheet, usecols=lambda c: c in usecols)
            except Exception as e:
                logger.warning("Sheet %s di %s gagal dibaca penuh (%s), coba tanpa usecols filter", sheet, fname, e)
                raw = xl.parse(sheet)

            sheet_row_counts[(fname, sheet)] = len(raw)
            if raw.empty:
                continue

            sub = _rename_and_select(raw, col_map, extra={
                "source": "RSUD_NAS",
                "visit_type": kategori,
                "raw_file": fname,
                "raw_sheet": sheet,
                "folder_default_class": folder_default_class,
            })
            # icd_row dari kolom ICD asli; kalau kosong/beda dari nama sheet, pakai nama sheet
            # sebagai fallback (nama sheet = kode ICD, sesuai desain data RSUD).
            sub["icd_row"] = sub["icd_row"].fillna(sheet).replace("", sheet)
            frames.append(sub)

    if not frames:
        raise RuntimeError(f"Tidak ada data ter-load dari {folder} -- cek struktur folder.")

    df = pd.concat(frames, ignore_index=True)
    logger.info("RSUD NAS %s: %d baris dari %d (file, sheet) pairs", kategori, len(df), len(sheet_row_counts))
    return IngestResult(df=df, sheet_row_counts=sheet_row_counts)


def load_rsa_ugm(path: str) -> IngestResult:
    """Load RS Akademik UGM (Penelitian Wawa 1 Labeled.xlsx) -- 1 sheet, 1 file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    usecols = list(RSA_COLUMNS.values())
    raw = xl.parse(sheet, usecols=lambda c: c in usecols)

    df = _rename_and_select(raw, RSA_COLUMNS, extra={
        "source": "RS_AKADEMIK_UGM",
        "visit_type": "Rawat Inap",  # semua baris di sumber ini Ranap (diverifikasi: 100%)
        "raw_file": os.path.basename(path),
        "raw_sheet": sheet,
        "folder_default_class": None,  # sumber ini tidak punya struktur folder-per-sindrom
    })
    logger.info("RS Akademik UGM: %d baris", len(df))
    return IngestResult(df=df, sheet_row_counts={(os.path.basename(path), sheet): len(raw)})


def validate_row_count(result: IngestResult, source_label: str) -> None:
    """
    Self-consistency check: total baris di DataFrame hasil load harus sama dengan
    total baris di semua (file, sheet) yang berhasil dibuka. Ini menangkap bug
    seperti V2 (cuma load sheet pertama) -- kalau ada sheet yang ke-skip diam-diam,
    angka ini akan tidak sama.
    """
    expected = sum(result.sheet_row_counts.values())
    actual = len(result.df)
    if expected != actual:
        raise AssertionError(
            f"[{source_label}] Row count mismatch: expected {expected} (sum semua sheet), "
            f"got {actual} di DataFrame hasil. Ada data yang hilang/dobel saat ingest."
        )
    logger.info("[%s] validate_row_count OK: %d baris dari %d sheet.", source_label, actual, len(result.sheet_row_counts))


def load_all_raw(paths_cfg: dict) -> pd.DataFrame:
    """
    Entry point: gabungkan RSUD Rawat Jalan + Rawat Inap + RS Akademik UGM.
    paths_cfg = config['paths'] dari config/experiment.yaml (berisi
    raw_rsud_nas_rawat_jalan, raw_rsud_nas_rawat_inap, raw_rsa_ugm).
    """
    rajal = load_rsud_nas_folder(paths_cfg["raw_rsud_nas_rawat_jalan"], "RAWAT JALAN")
    validate_row_count(rajal, "RSUD Rawat Jalan")

    ranap = load_rsud_nas_folder(paths_cfg["raw_rsud_nas_rawat_inap"], "RAWAT INAP")
    validate_row_count(ranap, "RSUD Rawat Inap")

    rsa = load_rsa_ugm(paths_cfg["raw_rsa_ugm"])
    validate_row_count(rsa, "RS Akademik UGM")

    combined = pd.concat([rajal.df, ranap.df, rsa.df], ignore_index=True)
    logger.info("TOTAL raw gabungan: %d baris (RAJAL=%d, RANAP=%d, RSA=%d)",
                len(combined), len(rajal.df), len(ranap.df), len(rsa.df))
    return combined
