"""
Baca experiments/runs.jsonl -> generate matriks perbandingan 6 iterasi
(data-v1/v2/v3 x model-base/large) + tabel per-kelas.

Jalankan: python -m reports.generate_report
Output  : reports/matriks_perbandingan.md, reports/per_kelas_<run_id>.md
"""

from __future__ import annotations

import json
import os

import pandas as pd

RUNS_PATH = "experiments/runs.jsonl"
OUT_DIR = "reports"

MATRIX_COLUMNS = [
    "run_id", "data_version", "model_key", "n_train", "n_val", "n_test", "num_classes",
    "val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro",
    "val_f1_macro_reliable_only", "val_f1_weighted",
    "test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro",
    "test_f1_macro_reliable_only", "test_f1_weighted",
    "training_time_sec", "smoke_test",
]


def load_runs(path: str = RUNS_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def build_matrix_table(runs: list[dict]) -> pd.DataFrame:
    if not runs:
        # kerangka kosong dgn 6 baris placeholder -- biar strukturnya kelihatan
        # walau belum ada run yang selesai (semua run butuh Kaggle GPU).
        placeholder_rows = []
        for dv in ["v1", "v2", "v3"]:
            for mk in ["base", "large"]:
                placeholder_rows.append({"run_id": f"{dv}_{mk}", "data_version": dv, "model_key": mk})
        df = pd.DataFrame(placeholder_rows)
    else:
        df = pd.DataFrame(runs)

    for col in MATRIX_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[MATRIX_COLUMNS]


def build_per_class_tables(runs: list[dict]) -> dict[str, pd.DataFrame]:
    tables = {}
    for r in runs:
        per_class = r.get("test_per_class")
        if not per_class:
            continue
        rows = [{"class": cls, **metrics} for cls, metrics in per_class.items()]
        tables[r["run_id"]] = pd.DataFrame(rows).sort_values("support", ascending=False)
    return tables


def to_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # fallback kalau tabulate belum terinstall
        return df.to_string(index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = load_runs()
    matrix = build_matrix_table(runs)

    lines = ["# Matriks Perbandingan -- 6 Iterasi (data-v1/v2/v3 x model-base/large)\n"]
    if not runs:
        lines.append("> Belum ada run yang selesai. Semua 6 kombinasi butuh training GPU "
                      "(jalankan di Kaggle -- lihat README.md). Tabel di bawah cuma kerangka.\n")
    lines.append(to_markdown_table(matrix))
    with open(os.path.join(OUT_DIR, "matriks_perbandingan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    per_class_tables = build_per_class_tables(runs)
    for run_id, df in per_class_tables.items():
        path = os.path.join(OUT_DIR, f"per_kelas_{run_id}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Performa per Kelas -- {run_id}\n\n" + to_markdown_table(df) + "\n")

    print(f"Ditulis: {OUT_DIR}/matriks_perbandingan.md ({len(runs)} run tercatat)")
    for run_id in per_class_tables:
        print(f"Ditulis: {OUT_DIR}/per_kelas_{run_id}.md")


if __name__ == "__main__":
    main()
