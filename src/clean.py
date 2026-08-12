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

import hashlib
import re
import logging
import unicodedata

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


# Marker ini bukan pengganti label ICD dan tidak dipakai untuk me-relabel data.
# Fungsinya hanya sebagai guard kualitas V4: catatan kontrol/teks sangat pendek
# tanpa satu pun sinyal yang selaras dengan kelasnya tidak cukup informatif untuk
# supervised learning. Polanya sengaja lebar agar false-drop tetap rendah.
V4_CLASS_ANCHOR_PATTERNS: dict[str, str] = {
    "Pneumonia/ISPA": r"batuk|pilek|sesak|napas|nafas|ronk|wheez|ispa|pneum|saturasi|spo2|dahak",
    "Suspek Dengue": r"dbd|dengue|trombosit|peteki|petech|mimisan|gusi berdarah|rumple|tourniquet|nyeri.{0,15}mata|demam",
    "COVID-19 Konfirmasi": r"covid|corona|pcr|antigen|swab|anosmia|ageusia|isoman|kontak erat|batuk|demam|sesak",
    "Diare Akut": r"diare|mencret|bab cair|berak cair|feses cair|muntah",
    "Diare Berdarah": r"diare|mencret|bab cair|berak cair|feses cair|darah|lendir",
    "Acute Flaccid Paralysis": r"lumpuh|kelemahan|layuh|kesemutan|kebas|baal|parese|paral|sulit berjalan|tidak bisa berjalan|tangan|kaki",
    "Sindrom Jaundice Akut": r"kuning|ikter|jaundice|hepatitis|bilirubin|hbsag",
    "Suspek HFMD": r"hfmd|hand foot|tangan.{0,30}kaki.{0,30}mulut|kaki.{0,30}tangan.{0,30}mulut|sariawan|vesikel|ruam|bintik",
    "Suspek Tetanus": r"tetanus|trismus|kaku|rahang|sulit menelan|kejang|tertusuk",
    "GHPR": r"gigit|digigit|gigitan|anjing|kucing|monyet|rabies|luka",
    "Suspek Leptospirosis": r"lepto|nyeri betis|mata merah|tikus|banjir|demam",
    "Suspek Meningitis/Ensefalitis": r"mening|ensefal|kaku kuduk|penurunan kesadaran|tidak sadar|kejang|sakit kepala|nyeri leher",
}

_V4_CONTROL_PATTERN = re.compile(
    r"\b(?:kontrol|post ranap|post mrs|post rawat|post opname|tanpa keluhan|"
    r"tidak ada keluhan|tdk ada keluhan)\b",
    re.IGNORECASE,
)


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


def normalize_anamnesa_v4(text) -> str:
    """Normalisasi teks konservatif untuk V4 tanpa menghapus informasi klinis.

    NFKC merapikan variasi Unicode, karakter kontrol dibuang, dan whitespace
    disatukan. Huruf/angka dan isi anamnesis tetap dipertahankan agar transform
    ini tidak mengubah makna klinis.
    """
    t = clean_anamnesa_text(text)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def canonicalize_anamnesa(text) -> str:
    """Representasi privacy-safe-in-memory untuk deteksi template/duplikat.

    Angka diganti token umum agar perbedaan tanggal/nilai lab saja tidak membuat
    dua template identik terlihat berbeda. Nilai ini hanya helper dan tidak
    disimpan ke CSV hasil.
    """
    t = normalize_anamnesa_v4(text).lower()
    t = re.sub(r"\d+(?:[.,]\d+)?", "<n>", t)
    t = re.sub(r"[^a-z<>]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_v4_class_anchor(text: str, class_name: str) -> bool:
    pattern = V4_CLASS_ANCHOR_PATTERNS.get(class_name)
    return bool(pattern and re.search(pattern, text.lower()))


def has_v4_clinical_signal(text: str) -> bool:
    """Deteksi sinyal klinis tanpa melihat label target baris."""
    normalized = text.lower()
    return _contains_any(normalized, _ACTIVE_SYMPTOM_KEYWORDS) or any(
        re.search(pattern, normalized)
        for pattern in V4_CLASS_ANCHOR_PATTERNS.values()
    )


def normalize_metadata_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Rapikan metadata kategorikal/numerik; nilai invalid menjadi missing."""
    out = df.copy()
    sex_map = {
        "L": "L", "M": "L", "MALE": "L", "LAKI-LAKI": "L", "LAKI LAKI": "L",
        "P": "P", "F": "P", "FEMALE": "P", "PEREMPUAN": "P", "WANITA": "P",
    }
    out["sex"] = out["sex"].apply(
        lambda value: sex_map.get(str(value).strip().upper()) if pd.notna(value) else None
    )
    out["age_years"] = pd.to_numeric(out["age_years"], errors="coerce")
    out.loc[~out["age_years"].between(0, 110), "age_years"] = None
    out["bulan_kunjung"] = pd.to_numeric(out["bulan_kunjung"], errors="coerce")
    out.loc[~out["bulan_kunjung"].between(1, 12), "bulan_kunjung"] = None
    out["visit_type"] = (
        out["visit_type"].astype(str).str.strip().str.upper()
        .replace({"RAWAT INAP": "RAWAT INAP", "RAWAT JALAN": "RAWAT JALAN"})
    )
    return out


def pseudonymize_record_ids(
    df: pd.DataFrame,
    record_id_col: str = "record_id",
    source_col: str = "source",
) -> pd.DataFrame:
    """Ganti identifier sumber dengan SHA-256 sebelum data V4 ditulis ke disk.

    Hash mencakup source agar namespace nomor pasien antar-RS tidak tertukar.
    Ini pseudonymization, bukan anonymization: data tetap harus diperlakukan
    sensitif dan dataset Kaggle tetap wajib private.
    """
    out = df.copy()

    def digest(source, record_id):
        if pd.isna(record_id) or str(record_id).strip().lower() in {"", "nan", "none"}:
            return None
        payload = f"{source}\x1f{str(record_id).strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    out[record_id_col] = [
        digest(source, record_id)
        for source, record_id in zip(out[source_col], out[record_id_col])
    ]
    return out


def apply_v4_quality_filters(
    df: pd.DataFrame,
    text_col: str = "anamnesa",
    class_col: str = "final_class",
    record_id_col: str = "record_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Terapkan filter V4 dan return ``(kept, aggregate_audit)``.

    Tidak ada relabel atau selection berbasis keyword kelas. Baris hanya dibuang untuk failure mode
    yang bisa diaudit: teks kosong, teks <=2 token tanpa anchor kelas, catatan
    kontrol tanpa anchor kelas, template yang identik tetapi berlabel konflik,
    atau kunjungan pasien yang identik berulang. Audit hanya berisi agregat;
    teks dan record_id pasien tidak pernah ditulis ke report.
    """
    out = normalize_metadata_v4(df)
    out[text_col] = out[text_col].apply(normalize_anamnesa_v4)
    canonical = out[text_col].apply(canonicalize_anamnesa)
    word_count = canonical.str.split().str.len().fillna(0).astype(int)
    class_anchor_match = pd.Series(
        [has_v4_class_anchor(text, cls) for text, cls in zip(out[text_col], out[class_col])],
        index=out.index,
    )
    clinical_signal = out[text_col].apply(has_v4_clinical_signal)
    reason = pd.Series("", index=out.index, dtype="object")

    reason.loc[word_count.eq(0)] = "empty_text"
    reason.loc[reason.eq("") & word_count.le(2) & ~clinical_signal] = "short_text_without_clinical_signal"
    is_control = out[text_col].str.contains(_V4_CONTROL_PATTERN, na=False)
    reason.loc[reason.eq("") & is_control & ~clinical_signal] = "control_without_clinical_signal"

    eligible = reason.eq("")
    conflict_counts = out.loc[eligible].assign(_canonical=canonical[eligible]).groupby(
        "_canonical"
    )[class_col].nunique()
    conflicting_templates = set(conflict_counts[conflict_counts > 1].index)
    reason.loc[reason.eq("") & canonical.isin(conflicting_templates)] = "conflicting_template_labels"

    # Hanya dedup pengulangan pasien+kelas+teks yang sama. Template identik pada
    # pasien berbeda tetap dipertahankan, tetapi split V4 menyatukannya ke group
    # yang sama agar tidak bocor antar train/val/test.
    eligible = reason.eq("")
    duplicate_visit = out.loc[eligible].assign(_canonical=canonical[eligible]).duplicated(
        [record_id_col, class_col, "_canonical"], keep="first"
    )
    reason.loc[duplicate_visit[duplicate_visit].index] = "duplicate_patient_visit"

    audit_base = out[[class_col, "source"]].copy()
    audit_base["reason"] = reason.replace("", "kept")
    audit = (
        audit_base.groupby(["reason", class_col, "source"], dropna=False)
        .size().reset_index(name="n_rows")
        .sort_values(["reason", "n_rows"], ascending=[True, False])
        .reset_index(drop=True)
    )

    kept = out.loc[reason.eq("")].copy().reset_index(drop=True)
    # Diagnostic-only: tidak pernah dipakai untuk memutuskan baris kept/drop.
    kept["v4_anchor_match"] = class_anchor_match.loc[reason.eq("")].to_numpy()
    kept["v4_word_count"] = word_count.loc[reason.eq("")].to_numpy()
    logger.info(
        "apply_v4_quality_filters: %d -> %d baris; drop=%s",
        len(out), len(kept),
        reason[reason.ne("")].value_counts().to_dict(),
    )
    return kept, audit


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
