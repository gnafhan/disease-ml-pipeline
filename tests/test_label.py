import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label import map_icd_to_class, normalize_icd


def test_normalize_icd_handles_comma_and_case():
    assert normalize_icd("b08,4") == "B08.4"
    assert normalize_icd(" a90 ") == "A90"
    assert normalize_icd(None) == ""


def test_dengue_mapping():
    assert map_icd_to_class("A90") == "Suspek Dengue"
    assert map_icd_to_class("A91.9") == "Suspek Dengue"


def test_covid_mapping():
    assert map_icd_to_class("U07.1") == "COVID-19 Konfirmasi"
    assert map_icd_to_class("U07.2") == "COVID-19 Konfirmasi"


def test_hfmd_specific_before_generic_b08():
    # B08.4 harus ke HFMD, TAPI B08.2 (Roseola) harus TIDAK ke-mapping sama sekali
    assert map_icd_to_class("B08.4") == "Suspek HFMD"
    assert map_icd_to_class("B08,4") == "Suspek HFMD"
    assert map_icd_to_class("B08.2") is None


def test_pneumonia_and_ispa_merge_into_same_class():
    assert map_icd_to_class("J00") == "Pneumonia/ISPA"
    assert map_icd_to_class("J18.0") == "Pneumonia/ISPA"
    assert map_icd_to_class("J18.9") == "Pneumonia/ISPA"


def test_afp_covers_neuro_codes_not_just_polio():
    assert map_icd_to_class("A80") == "Acute Flaccid Paralysis"
    assert map_icd_to_class("G56.0") == "Acute Flaccid Paralysis"
    assert map_icd_to_class("G83.1") == "Acute Flaccid Paralysis"


def test_jaundice_covers_r17_and_hepatitis():
    assert map_icd_to_class("R17") == "Sindrom Jaundice Akut"
    assert map_icd_to_class("B16.9") == "Sindrom Jaundice Akut"
    assert map_icd_to_class("B17.1") == "Sindrom Jaundice Akut"


def test_excluded_codes_return_none():
    for code in ["J09.8", "B50", "B54", "B05", "B06.9", "B00.5", "B20.9", "A38", "A01"]:
        assert map_icd_to_class(code) is None, f"{code} seharusnya TIDAK ke-mapping (di luar 12 kelas)"


def test_unmapped_code_returns_none():
    assert map_icd_to_class("Z99.9") is None
    assert map_icd_to_class("") is None


def test_multi_code_cell_picks_first_mappable_code():
    # kasus nyata: pasien 2 diagnosis, "J00,\n K30," (common cold + dyspepsia)
    assert map_icd_to_class("J00,\n K30,") == "Pneumonia/ISPA"
    # kode simptom umum (R50.9 fever unspecified) di depan, diagnosis definitif
    # (J00) di belakang -- harus tetap ketemu, bukan cuma ambil kode pertama
    assert map_icd_to_class("R50.9, J00") == "Pneumonia/ISPA"
    # comma-decimal (bukan multi-code) tetap harus ke-parse sbg SATU kode
    assert map_icd_to_class("b08,4") == "Suspek HFMD"


def test_multi_code_with_no_mappable_code_returns_none():
    assert map_icd_to_class("R50.9, K30") is None
