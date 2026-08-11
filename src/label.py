"""
Mapping ICD-10 (per baris) -> kelas final penyakit.

REKONSTRUKSI, bukan skrip asli -- catatan penting:
  Skrip yang dipakai untuk bikin DATASET_GABUNGAN_SPLIT_V4/V5b (versi "ICD-based"
  yang paling benar di riwayat project ini) TIDAK ketemu di notebook manapun yang
  tersisa (v2_v3.ipynb, v5_fixed.py, v6_training.py, v7_ensemble.py semuanya cuma
  MEMAKAI dataset yang sudah jadi, bukan proses bikinnya). Mapping di bawah ini
  disusun ulang dari 3 sumber:
    1. `ICD SKDR versi 30 Desember 2024 (1).xlsx` -- mapping resmi dokter
       (1 kode contoh per penyakit; beberapa kode contoh di file ini TIDAK
       cocok dengan kode yang benar-benar dipakai di data asli, lihat catatan
       per kelas di bawah).
    2. Inventarisasi kode ICD nyata di seluruh 25 file RSUD NAS + kolom `ICD 10`
       RS Akademik UGM (lihat hasil investigasi sesi ini).
    3. Cross-check ke kolom `sindrom` RS Akademik UGM (1857 baris) -- untuk
       SEMUA kelas kecuali "Suspek Campak", jumlah baris hasil mapping ICD di
       bawah ini MATCH PERSIS dengan hitungan sindrom aslinya (COVID 1113,
       AFP 94, Tetanus 24, HFMD 24, Leptospirosis 13, Jaundice 123, Diare
       Akut 7, Diare Berdarah 1, Malaria 3 -- semua exact match). Ini validasi
       kuat bahwa aturan prefix di bawah sudah benar.

  ACTION ITEM untuk TA: sanity-check tabel ini ke dr. Wawa / dosen pembimbing
  sebelum treat hasil training sebagai final -- terutama untuk kelas yang
  kodenya BEDA dari tabel resmi (Diare Akut, Sindrom Jaundice Akut, AFP).

Catatan per kelas (kode resmi di tabel dokter vs kode nyata di data):
  - Diare Akut: tabel resmi bilang A02 (Salmonella), tapi SEMUA data RSUD NAS &
    RS Akademik pakai A09.x (diare infeksius). Dipakai: A09.
  - Sindrom Jaundice Akut: tabel resmi bilang A95 (Yellow Fever, tidak endemik
    di Indonesia -- kemungkinan cuma placeholder). Data nyata pakai R17
    (jaundice unspecified) & B15-B19 (hepatitis virus). Dipakai: R17 + B15-B19.
  - Acute Flaccid Paralysis: tabel resmi bilang A80 (polio). Data nyata isinya
    hampir semua kode neuromuskular G5x/G6x/G7x/G8x (carpal tunnel, GBS,
    miopati, dll). Ini SESUAI standar surveilans AFP WHO -- program AFP
    memang menangkap semua kasus lumpuh layuh akut, mayoritas ujungnya BUKAN
    polio (non-polio AFP). Dipakai: G5x-G8x + A80.
  - "Suspek Campak" DIHAPUS TOTAL dari taksonomi kelas final (bukan cuma
    di-drop belakangan) -- sesuai temuan PLAN_V6_DATA_FIX.md bahwa label ini
    95% sebenarnya Dengue yang salah kategori surveilans, sisanya penyakit lain
    yang tidak relevan (HIV, Herpes, dll). ICD asli di baris-baris itu (A90,
    B16.9, dst) tetap di-reclass ke kelas yang benar lewat aturan di bawah.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ── Aturan ICD prefix -> kelas final. Semua kelas di sini SAMA PERSIS dengan  ──
# ── daftar `classes` di config/experiment.yaml (12 kelas, termasuk           ──
# ── 'Pneumonia/ISPA' yang sudah digabung) -- dipakai sama untuk v1/v2/v3      ──
# ── supaya lintas versi data tetap bisa dibandingkan head-to-head.           ──
#
# Urutan penting: prefix lebih panjang/spesifik dicek duluan (mis. B08.4
# sebelum B08 umum) supaya tidak ke-capture aturan yang lebih pendek/luas.

ICD_PREFIX_RULES: list[tuple[str, str]] = [
    # -- Dengue --
    ("A90", "Suspek Dengue"),
    ("A91", "Suspek Dengue"),
    # -- COVID-19 --
    ("U07", "COVID-19 Konfirmasi"),
    # -- Diare --
    ("A09", "Diare Akut"),
    ("A06", "Diare Berdarah"),
    # -- Acute Flaccid Paralysis (surveilans AFP, bukan cuma polio -- lihat docstring) --
    ("A80", "Acute Flaccid Paralysis"),
    ("G54", "Acute Flaccid Paralysis"),
    ("G56", "Acute Flaccid Paralysis"),
    ("G57", "Acute Flaccid Paralysis"),
    ("G61", "Acute Flaccid Paralysis"),
    ("G70", "Acute Flaccid Paralysis"),
    ("G71", "Acute Flaccid Paralysis"),
    ("G72", "Acute Flaccid Paralysis"),
    ("G81", "Acute Flaccid Paralysis"),
    ("G82", "Acute Flaccid Paralysis"),
    ("G83", "Acute Flaccid Paralysis"),
    # -- Sindrom Jaundice Akut (R17 jaundice + B15-B19 hepatitis virus) --
    ("R17", "Sindrom Jaundice Akut"),
    ("B15", "Sindrom Jaundice Akut"),
    ("B16", "Sindrom Jaundice Akut"),
    ("B17", "Sindrom Jaundice Akut"),
    ("B18", "Sindrom Jaundice Akut"),
    ("B19", "Sindrom Jaundice Akut"),
    # -- HFMD (harus dicek SEBELUM aturan umum lain, prefix spesifik B08.4) --
    ("B08.4", "Suspek HFMD"),
    ("B08,4", "Suspek HFMD"),  # varian koma yang ditemukan di beberapa sheet
    # -- Tetanus --
    ("A35", "Suspek Tetanus"),
    ("A33", "Suspek Tetanus"),
    # -- GHPR (gigitan hewan penular rabies) --
    ("W55", "GHPR"),
    # -- Leptospirosis --
    ("A27", "Suspek Leptospirosis"),
    # -- Meningitis/Ensefalitis --
    ("G00", "Suspek Meningitis/Ensefalitis"),
    ("G01", "Suspek Meningitis/Ensefalitis"),
    ("G02", "Suspek Meningitis/Ensefalitis"),
    ("G03", "Suspek Meningitis/Ensefalitis"),
    ("G04", "Suspek Meningitis/Ensefalitis"),
    ("G05", "Suspek Meningitis/Ensefalitis"),
    ("G06", "Suspek Meningitis/Ensefalitis"),
    ("G07", "Suspek Meningitis/Ensefalitis"),
    ("G08", "Suspek Meningitis/Ensefalitis"),
    ("G09", "Suspek Meningitis/Ensefalitis"),
    ("A88", "Suspek Meningitis/Ensefalitis"),
    ("A83", "Suspek Meningitis/Ensefalitis"),
    # -- ISPA (uri) & Pneumonia -- DIGABUNG langsung jadi 'Pneumonia/ISPA' di sini
    # (bukan langkah terpisah di clean.py) supaya taksonomi 12 kelas di
    # config/experiment.yaml berlaku SAMA persis untuk data-v1/v2/v3 -- sesuai
    # keputusan "classes: dipakai untuk MENYARING semua versi data" yang sudah
    # dikunci di config. Dari anamnesa saja, Pneumonia vs ISPA tidak bisa
    # dibedakan tanpa foto toraks (F1 Pneumonia=0.00 di semua versi lama).
    ("J00", "Pneumonia/ISPA"), ("J01", "Pneumonia/ISPA"), ("J02", "Pneumonia/ISPA"),
    ("J03", "Pneumonia/ISPA"), ("J04", "Pneumonia/ISPA"), ("J05", "Pneumonia/ISPA"),
    ("J06", "Pneumonia/ISPA"), ("J20", "Pneumonia/ISPA"), ("J21", "Pneumonia/ISPA"),
    ("J22", "Pneumonia/ISPA"),
    ("J12", "Pneumonia/ISPA"), ("J13", "Pneumonia/ISPA"), ("J14", "Pneumonia/ISPA"),
    ("J15", "Pneumonia/ISPA"), ("J16", "Pneumonia/ISPA"), ("J17", "Pneumonia/ISPA"),
    ("J18", "Pneumonia/ISPA"),
]

# Kode yang SENGAJA di-drop (bukan salah baca, tapi memang di luar scope 12 kelas
# final -- didokumentasikan biar jelas kenapa, bukan silent drop):
KNOWN_EXCLUDED_PREFIXES = {
    "J09": "Influenza/ILI -- terlalu sedikit sampel (1 baris), tidak masuk taksonomi final",
    "J10": "Influenza -- idem",
    "J11": "Influenza -- idem",
    "B50": "Malaria -- terlalu sedikit & tidak masuk taksonomi final",
    "B51": "Malaria -- idem",
    "B52": "Malaria -- idem",
    "B53": "Malaria -- idem",
    "B54": "Malaria -- idem",
    "B05": "Campak (Measles) -- kelas Suspek Campak dihapus total, lihat docstring modul",
    "B06": "Rubella/Campak Jerman -- idem",
    "B00": "Herpes simplex -- tidak relevan, ditemukan nyasar di folder Suspek Campak",
    "B20": "HIV -- tidak relevan, ditemukan nyasar di folder Suspek Campak",
    "A38": "Scarlet fever -- tidak relevan",
    "A01": "Demam Tifoid -- terlalu sedikit sampel, tidak masuk taksonomi final",
    "B08.2": "Roseola/Exanthema subitum -- tidak relevan",
    "R50.2": "Drug-induced fever -- tidak relevan",
}


_ICD_TOKEN_PATTERN = re.compile(r"[A-Z]\d{2}(?:[.,]\d{1,2})?")


def normalize_icd(raw) -> str:
    """Normalisasi SATU kode ICD tunggal (bukan cell mentah yang mungkin
    berisi beberapa kode -- lihat extract_icd_tokens untuk itu)."""
    if pd.isna(raw):
        return ""
    s = str(raw).strip().upper()
    s = s.replace(",", ".")
    s = re.sub(r"\s+", "", s)
    return s


def extract_icd_tokens(raw) -> list[str]:
    """
    Satu cell 'Kode ICD'/'ICD Code'/'ICD 10' KADANG berisi >1 kode -- pasien
    dengan diagnosis ganda, mis. "J00,\\n K30," (common cold + dyspepsia).
    Regex ini menangkap tiap kode individual dengan benar SEKALIGUS
    membedakannya dari notasi desimal-koma tunggal (mis. sheet name "b08,4"
    yang artinya B08.4, bukan dua kode terpisah) -- koma yang langsung
    diikuti 1-2 digit dianggap titik desimal punya token yg sama; koma yang
    diikuti spasi/baris baru + huruf baru dianggap PEMISAH ke kode lain.
    """
    if pd.isna(raw):
        return []
    s = str(raw).upper()
    return _ICD_TOKEN_PATTERN.findall(s)


def map_icd_to_class(icd_cell) -> str | None:
    """
    Terima 1 kode ATAU 1 cell mentah yang mungkin berisi beberapa kode.
    Kalau ada >1 kode dalam satu cell, kode PERTAMA yang berhasil ke-mapping
    ke salah satu dari 12 kelas final yang dipakai (bukan otomatis kode
    pertama dalam urutan penulisan -- kode pertama seringkali cuma simptom
    umum spt R50.9 'fever unspecified', bukan diagnosis definitif).
    """
    for token in extract_icd_tokens(icd_cell):
        code = normalize_icd(token)
        for prefix, cls in sorted(ICD_PREFIX_RULES, key=lambda x: -len(x[0])):
            if code.startswith(prefix):
                return cls
    return None


def apply_labels(df: pd.DataFrame, icd_col: str = "icd_row") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tambah kolom 'final_class'. Baris yang ICD-nya tidak ke-mapping (di luar 12+1
    taksonomi v1) DIBUANG dari df utama, tapi diringkas ke drop_summary (agregat
    per isi cell ICD mentah -- TIDAK mengandung teks anamnesa, aman untuk
    di-log/disimpan) supaya keputusan "buang" ini auditable, bukan silent drop.
    """
    df = df.copy()
    df["final_class"] = df[icd_col].apply(map_icd_to_class)
    df["_icd_norm"] = df[icd_col].apply(lambda v: "" if pd.isna(v) else re.sub(r"\s+", " ", str(v).strip()))

    mapped = df[df["final_class"].notna()].drop(columns=["_icd_norm"])
    unmapped = df[df["final_class"].isna()]

    if len(unmapped):
        drop_summary = (
            unmapped.groupby("_icd_norm")
            .size()
            .reset_index(name="n_dropped")
            .sort_values("n_dropped", ascending=False)
        )
        drop_summary["known_reason"] = drop_summary["_icd_norm"].apply(
            lambda c: next((v for k, v in KNOWN_EXCLUDED_PREFIXES.items() if c.startswith(k)), "UNKNOWN -- perlu dicek manual")
        )
    else:
        drop_summary = pd.DataFrame(columns=["_icd_norm", "n_dropped", "known_reason"])

    n_unknown = (drop_summary["known_reason"] == "UNKNOWN -- perlu dicek manual").sum() if len(drop_summary) else 0
    if n_unknown:
        logger.warning(
            "%d kode ICD unik ke-drop TANPA alasan terdokumentasi -- cek drop_summary, "
            "mungkin ada kelas yang belum ke-cover aturan mapping.", n_unknown
        )
    logger.info("apply_labels: %d baris berlabel, %d baris dibuang (ICD tidak masuk taksonomi 12 kelas)",
                len(mapped), len(unmapped))
    return mapped, drop_summary
