import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.split import stratified_group_split


def _fixture_df():
    rows = []
    for cls in ["A", "B", "C"]:
        for patient in range(20):
            # Dua kunjungan per pasien. Template tertentu juga sengaja dipakai
            # pasien berbeda untuk menguji connected patient+text grouping.
            template = f"keluhan {cls} template {patient % 7}"
            for visit in range(2):
                rows.append({
                    "record_id": f"{cls}-{patient}",
                    "anamnesa": template if visit == 0 else f"{template} followup",
                    "final_class": cls,
                    "source": "RS1" if patient % 2 else "RS2",
                })
    return pd.DataFrame(rows)


def test_v4_group_split_preserves_every_row_and_prevents_patient_leakage():
    df = _fixture_df()
    train, val, test = stratified_group_split(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)

    id_sets = [set(part["record_id"]) for part in (train, val, test)]
    assert not (id_sets[0] & id_sets[1])
    assert not (id_sets[0] & id_sets[2])
    assert not (id_sets[1] & id_sets[2])


def test_v4_group_split_keeps_identical_templates_in_one_subset():
    df = _fixture_df()
    parts = stratified_group_split(df, seed=42)
    seen = {}
    for split_name, part in zip(["train", "val", "test"], parts):
        for text in part["anamnesa"]:
            seen.setdefault(text, set()).add(split_name)
    assert all(len(splits) == 1 for splits in seen.values())


def test_v4_group_split_is_reproducible():
    df = _fixture_df()
    first = stratified_group_split(df, seed=42)
    second = stratified_group_split(df, seed=42)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
