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


_CANONICAL_RUN_ORDER = [f"{dv}_{mk}" for dv in ("v1", "v2", "v3") for mk in ("base", "large")]


def load_runs(path: str = RUNS_PATH) -> list[dict]:
    """
    Baca experiments/runs.jsonl. Satu run_id (mis. "v1_base") bisa muncul lebih
    dari sekali di file ini -- misal smoke-test dulu (--smoke-test) baru
    training asli, atau resume sesi Kaggle yang keputus lalu diulang. Dalam
    kasus itu, entri YANG PALING TERAKHIR ditulis (baris paling bawah) yang
    dipakai -- itu representasi paling baru/final utk run_id tersebut, entri
    lama (mis. hasil smoke-test dgn f1=0.0) TIDAK boleh ikut nongol di
    matriks final.
    """
    if not os.path.exists(path):
        return []
    by_run_id: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            run_id = record.get("run_id")
            if run_id:
                by_run_id[run_id] = record  # overwrite -> entri terakhir menang

    # urutan stabil: kombinasi kanonik (v1_base..v3_large) dulu, run_id asing
    # (kalau ada) ditaruh di belakang, bukan hilang diam-diam.
    ordered = [by_run_id[rid] for rid in _CANONICAL_RUN_ORDER if rid in by_run_id]
    leftover = [r for rid, r in by_run_id.items() if rid not in _CANONICAL_RUN_ORDER]
    return ordered + leftover


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
