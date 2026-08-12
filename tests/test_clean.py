import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import (
    clean_anamnesa_text, remove_control_visits_without_complaint,
    remove_incidental_covid, flag_unreliable_classes,
    normalize_anamnesa_v4, normalize_metadata_v4, apply_v4_quality_filters,
    has_v4_clinical_signal, pseudonymize_record_ids,
)


def test_clean_anamnesa_text_strips_soap_headers():
    raw = "KU: demam\n\nRPS:\nPasien mengeluh demam 3 hari"
    cleaned = clean_anamnesa_text(raw)
    assert "KU:" not in cleaned
    assert "RPS:" not in cleaned
    assert "demam" in cleaned


def test_control_visit_without_complaint_is_dropped():
    df = pd.DataFrame({
        "anamnesa": [
            "kontrol bulan ini tdk ada keluhan. anak aktif",
            "kontrol saat ini batuk sejak kemarin, sesak (-)",
            "demam sejak 3 hari, batuk pilek",
        ]
    })
    result = remove_control_visits_without_complaint(df)
    assert len(result) == 2  # baris 1 (kontrol tanpa keluhan) dibuang
    assert "tdk ada keluhan" not in " ".join(result["anamnesa"])


def test_incidental_covid_is_dropped_but_symptomatic_covid_kept():
    df = pd.DataFrame({
        "anamnesa": [
            "Pasien G3P2 hamil 41 minggu rujukan dengan confirm covid by antigen. kenceng-kenceng.",
            "Demam, batuk, sesak nafas, anosmia sejak 5 hari, riwayat kontak covid",
        ],
        "final_class": ["COVID-19 Konfirmasi", "COVID-19 Konfirmasi"],
    })
    result = remove_incidental_covid(df)
    assert len(result) == 1
    assert "anosmia" in result["anamnesa"].iloc[0]


def test_flag_unreliable_classes():
    df = pd.DataFrame({"final_class": ["A"] * 50 + ["B"] * 5})
    result = flag_unreliable_classes(df, min_support=30)
    assert result[result["final_class"] == "A"]["is_reliable_class"].all()
    assert not result[result["final_class"] == "B"]["is_reliable_class"].any()


def test_normalize_anamnesa_v4_removes_control_chars_but_keeps_clinical_content():
    assert normalize_anamnesa_v4("KU:\x00  demam\n  3 hari") == "demam 3 hari"


def test_normalize_metadata_v4_invalid_values_become_missing():
    df = pd.DataFrame({
        "sex": ["male", "P", "22"],
        "age_years": [25, 999, -1],
        "bulan_kunjung": [1, 13, 0],
        "visit_type": ["Rawat Inap", "RAWAT JALAN", "rawat inap"],
    })
    result = normalize_metadata_v4(df)
    assert result["sex"].tolist()[:2] == ["L", "P"]
    assert pd.isna(result["sex"].iloc[2])
    assert pd.isna(result["age_years"].iloc[1])
    assert pd.isna(result["bulan_kunjung"].iloc[2])
    assert set(result["visit_type"]) == {"RAWAT INAP", "RAWAT JALAN"}


def test_v4_quality_filter_is_auditable_and_does_not_relabel():
    df = pd.DataFrame({
        "anamnesa": [
            "",  # kosong
            "kontrol post ranap",  # tidak punya sinyal klinis
            "demam trombosit turun",  # selaras dengan dengue
            "demam trombosit turun",  # kunjungan identik pasien yang sama
            "digigit kucing pada tangan",  # GHPR valid
        ],
        "final_class": ["Suspek Dengue", "Suspek Dengue", "Suspek Dengue", "Suspek Dengue", "GHPR"],
        "record_id": ["a", "b", "c", "c", "d"],
        "source": ["RSUD_NAS"] * 5,
        "sex": ["L"] * 5,
        "age_years": [20] * 5,
        "bulan_kunjung": [1] * 5,
        "visit_type": ["RAWAT JALAN"] * 5,
    })
    kept, audit = apply_v4_quality_filters(df)
    assert kept["final_class"].tolist() == ["Suspek Dengue", "GHPR"]
    counts = audit.groupby("reason")["n_rows"].sum().to_dict()
    assert counts["empty_text"] == 1
    assert counts["control_without_clinical_signal"] == 1
    assert counts["duplicate_patient_visit"] == 1


def test_v4_clinical_signal_does_not_depend_on_target_class():
    assert has_v4_clinical_signal("kontrol, kadang pusing")
    assert has_v4_clinical_signal("demam")
    assert not has_v4_clinical_signal("kontrol post ranap")


def test_v4_record_ids_are_pseudonymized_before_persistence():
    source = pd.DataFrame({
        "record_id": ["RM-123", "RM-123", "RM-123", None],
        "source": ["RS-A", "RS-A", "RS-B", "RS-A"],
    })
    result = pseudonymize_record_ids(source)
    assert result["record_id"].iloc[0] == result["record_id"].iloc[1]
    assert result["record_id"].iloc[0] != result["record_id"].iloc[2]
    assert len(result["record_id"].iloc[0]) == 64
    assert "RM-123" not in set(result["record_id"].dropna())
    assert pd.isna(result["record_id"].iloc[3])
