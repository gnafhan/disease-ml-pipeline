"""
Test dedup logic reports/generate_report.py::load_runs -- ini reaksi dari
kasus nyata: run --smoke-test dulu (nyimpen 6 record f1=0.0 ke runs.jsonl),
lalu run asli tanpa --smoke-test buat run_id yang sama. Karena runs.jsonl
ditulis dgn append (bukan overwrite), file itu jadi punya 2 baris per
run_id. matriks_perbandingan.md HARUS pakai entri yang PALING BARU
(training asli), bukan entri smoke-test yang lama, dan HARUS tidak
duplikat baris utk run_id yang sama.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reports.generate_report import build_matrix_table, load_runs

RUNS_PATH = "experiments/runs.jsonl"


def _write_runs(records):
    os.makedirs("experiments", exist_ok=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def setup_function(_):
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)


def teardown_function(_):
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)


def test_later_entry_for_same_run_id_wins_over_earlier_smoke_test():
    _write_runs([
        {"run_id": "v1_base", "smoke_test": True, "test_f1_macro": 0.0},
        {"run_id": "v1_large", "smoke_test": True, "test_f1_macro": 0.0},
        {"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.71},
    ])
    runs = load_runs()
    by_id = {r["run_id"]: r for r in runs}

    assert len(runs) == 2, "v1_base cuma boleh muncul SEKALI (deduped), bukan dobel"
    assert by_id["v1_base"]["smoke_test"] is False
    assert by_id["v1_base"]["test_f1_macro"] == 0.71
    assert by_id["v1_large"]["smoke_test"] is True  # belum ada training asli utk ini


def test_matrix_table_has_no_duplicate_run_id_rows():
    _write_runs([
        {"run_id": "v2_large", "smoke_test": True, "test_f1_macro": 0.0},
        {"run_id": "v2_large", "smoke_test": False, "test_f1_macro": 0.65},
        {"run_id": "v2_large", "smoke_test": False, "test_f1_macro": 0.68},  # re-run kedua
    ])
    runs = load_runs()
    matrix = build_matrix_table(runs)

    assert (matrix["run_id"] == "v2_large").sum() == 1
    assert matrix.loc[matrix["run_id"] == "v2_large", "test_f1_macro"].iloc[0] == 0.68


def test_canonical_order_preserved_regardless_of_write_order():
    _write_runs([
        {"run_id": "v3_large", "test_f1_macro": 0.5},
        {"run_id": "v1_base", "test_f1_macro": 0.6},
    ])
    runs = load_runs()
    assert [r["run_id"] for r in runs] == ["v1_base", "v3_large"]
