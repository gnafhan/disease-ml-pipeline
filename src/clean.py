"""
Pembersihan baris-level setelah labeling ICD (label.py).

Urutan pemakaian utk versioning 2-axis (lihat config/experiment.yaml & README):
  data-v1 = ingest + label saja (fungsi di modul ini belum dipakai)
  data-v2 = v1 + clean_anamnesa_text + remove_control_visits_without_complaint
            + remove_incidental_covid
  data-v3 = v2 + merge_pneumonia_into_ispa (+ flag_unreliable_classes utk reporting,
            TIDAK menghapus baris -- lihat catatan di bawah)

Semua aturan di bawah direkonstruksi dari REPORT_V6_FINAL.md ("Kenapa data
kontrol/post ranap dihapus?", "Kenapa COVID incidental dihapus?").
"""

from __future__ import annotations

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

SOAP_HEADER_PATTERN = re.compile(
    r"\b(KU|RPS|RPD|RPO|RO|RPK|Alergi)\s*:", re.IGNORECASE
)

_ACTIVE_SYMPTOM_KEYWORDS = [
    "batuk", "demam", "pilek", "sesak", "nyeri", "diare", "mencret", "muntah",
    "mual", "pusing", "lemas", "ruam", "gatal", "bab cair", "bak", "kejang",
    "sakit", "panas", "pendarahan", "berdarah", "bengkak", "kaku", "gigit",
    "digigit", "luka", "trombosit", "sesak nafas", "sesak napas",
]

_CONTROL_MARKERS = [
    "kontrol", "post ranap", "post mrs", "post rawat", "tidak ada keluhan",
    "tdk ada keluhan", "tanpa keluhan",
]

_OBSTETRIC_SURGICAL_MARKERS = [
    "hamil", "persalinan", "kenceng", "g1p0", "g2p1", "g3p2", "inpartu",
    "post op", "post sc", "bedah", "operasi", "kuretase", "curettage",
    "sectio", "partus",
]

_COVID_RELEVANT_MARKERS = [
    # SENGAJA tidak termasuk 'antigen'/'swab'/'pcr' -- itu metode tes, bukan
    # gejala, dan hampir selalu muncul di semua baris COVID (karena begitulah
    # cara dikonfirmasi) sehingga kalau dimasukkan sini bikin deteksi
    # "incidental" nyaris tidak pernah kena (lihat contoh di REPORT_V6_FINAL.md:
    # "confirm covid by antigen" pada pasien bersalin TETAP harus terdeteksi
    # sebagai incidental, walau ada kata "antigen").
    "anosmia", "ageusia", "sesak", "batuk", "pilek", "demam", "saturasi",
    "spo2", "isoman", "gejala covid", "kontak covid",
]


def clean_anamnesa_text(text) -> str:
    """Buang header SOAP (KU:/RPS:/RPD:/dst) dari teks Subjective RS Akademik.
    Aman dijalankan di teks RSUD juga -- kalau tidak ada header, teks dikembalikan
    apa adanya (cuma di-strip whitespace)."""
    if pd.isna(text):
        return ""
    t = str(text)
    t = SOAP_HEADER_PATTERN.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def remove_control_visits_without_complaint(df: pd.DataFrame, text_col: str = "anamnesa") -> pd.DataFrame:
    """
    Buang baris kontrol/post-ranap yang TIDAK ada keluhan aktif.
    Baris "kontrol" yang masih menyebutkan keluhan (mis. "kontrol, batuk masih
    ada") DIPERTAHANKAN -- itu masih informatif untuk model.
    """
    text = df[text_col].fillna("").astype(str)
    is_control = text.apply(lambda t: _contains_any(t, _CONTROL_MARKERS))
    has_active_complaint = text.apply(lambda t: _contains_any(t, _ACTIVE_SYMPTOM_KEYWORDS))

    drop_mask = is_control & ~has_active_complaint
    n_dropped = int(drop_mask.sum())
    logger.info("remove_control_visits_without_complaint: %d baris dibuang dari %d", n_dropped, len(df))
    return df[~drop_mask].reset_index(drop=True)


def remove_incidental_covid(df: pd.DataFrame, text_col: str = "anamnesa", class_col: str = "final_class") -> pd.DataFrame:
    """
    Buang baris berlabel COVID-19 Konfirmasi yang sebenarnya datang untuk
    persalinan/bedah (kebetulan antigen/PCR positif saat skrining masuk RS),
    BUKAN karena bergejala COVID.
    """
    text = df[text_col].fillna("").astype(str)
    is_covid = df[class_col] == "COVID-19 Konfirmasi"
    is_obstetric_surgical = text.apply(lambda t: _contains_any(t, _OBSTETRIC_SURGICAL_MARKERS))
    has_covid_symptom = text.apply(lambda t: _contains_any(t, _COVID_RELEVANT_MARKERS))

    drop_mask = is_covid & is_obstetric_surgical & ~has_covid_symptom
    n_dropped = int(drop_mask.sum())
    logger.info("remove_incidental_covid: %d baris dibuang dari %d", n_dropped, len(df))
    return df[~drop_mask].reset_index(drop=True)


def merge_pneumonia_into_ispa(df: pd.DataFrame, class_col: str = "final_class") -> pd.DataFrame:
    """
    LEGACY / tidak dipakai di pipeline.py saat ini -- label.py sekarang langsung
    memetakan kode ICD Pneumonia (J12-J18) ke 'Pneumonia/ISPA' sejak label stage,
    supaya taksonomi 12 kelas identik di data-v1/v2/v3 (lihat config.yaml).
    Fungsi ini dipertahankan sebagai dokumentasi/fallback kalau suatu saat label
    dipisah lagi jadi Pneumonia vs ISPA.

    Gabung kelas 'Pneumonia' + 'ISPA' -> 'Pneumonia/ISPA'.
    Alasan (REPORT_V6_FINAL.md): F1 Pneumonia = 0.00 di semua versi sebelumnya --
    dari anamnesa saja tidak bisa dibedakan dari ISPA tanpa foto toraks/auskultasi.
    """
    df = df.copy()
    n_affected = int(df[class_col].isin(["Pneumonia", "ISPA"]).sum())
    df[class_col] = df[class_col].replace({"Pneumonia": "Pneumonia/ISPA", "ISPA": "Pneumonia/ISPA"})
    logger.info("merge_pneumonia_into_ispa: %d baris di-relabel jadi 'Pneumonia/ISPA'", n_affected)
    return df


def flag_unreliable_classes(df: pd.DataFrame, class_col: str = "final_class", min_support: int = 30) -> pd.DataFrame:
    """
    TIDAK menghapus kelas langka (beda dari pendekatan V6 lama yang hapus
    Malaria/Campak/Tifoid/ILI langsung) -- kelas dengan total sampel < min_support
    cuma DITANDAI lewat kolom boolean 'is_reliable_class'. evaluate.py memakai
    flag ini untuk hitung macro_f1_reliable_only terpisah dari macro_f1_all,
    supaya kelas langka tetap kelihatan di laporan (bukan didiamkan hilang)
    tapi tidak menekan metrik utama secara tidak proporsional.
    """
    df = df.copy()
    counts = df[class_col].value_counts()
    reliable_classes = set(counts[counts >= min_support].index)
    df["is_reliable_class"] = df[class_col].isin(reliable_classes)

    n_unreliable_classes = counts[counts < min_support]
    if len(n_unreliable_classes):
        logger.info(
            "flag_unreliable_classes: %d kelas di bawah ambang %d sampel (TIDAK dihapus): %s",
            len(n_unreliable_classes), min_support,
            ", ".join(f"{cls}={n}" for cls, n in n_unreliable_classes.items()),
        )
    return df
