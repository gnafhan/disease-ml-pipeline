import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import (
    clean_anamnesa_text, remove_control_visits_without_complaint,
    remove_incidental_covid, flag_unreliable_classes,
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
