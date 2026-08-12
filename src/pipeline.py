"""
Orkestrasi tahap data: Ingest -> Label -> Clean -> Split -> simpan ke
data/processed/<version>/{train,val,test}.csv.

Dipanggil sbg:  python -m src.pipeline --data-version v1
                python -m src.pipeline --data-version v2
                python -m src.pipeline --data-version v3
                python -m src.pipeline --data-version v4

v1 = ingest + label saja (ISPA & Pneumonia masih terpisah, noise blm dibersihkan)
v2 = v1 + remove_control_visits_without_complaint + remove_incidental_covid
v3 = v2 + merge_pneumonia_into_ispa + flag_unreliable_classes (kelas final = 12,
     sama seperti daftar 'classes' di config/experiment.yaml)
v4 = v3 + filter kualitas berbasis bukti + normalisasi metadata + split grouped
     by pasien/template. Taksonomi tetap 12 kelas; tidak ada kelas langka yang
     dihapus untuk menaikkan skor.

Training (src/train.py) TIDAK dipanggil dari sini -- tahap training butuh GPU
(Kaggle), jadi sengaja dipisah supaya tahap data (murah, CPU-only) bisa
dijalankan berkali-kali tanpa nunggu training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import pandas as pd

from src import ingest, label as label_mod, clean, split as split_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def log_run(record: dict, runs_path: str = "experiments/runs.jsonl") -> None:
    os.makedirs(os.path.dirname(runs_path), exist_ok=True)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(data_version: str, cfg: dict):
    """Return data, timing, ICD audit, V4 quality audit, and stage counts."""
    timing = {}
    stage_counts = {}
    quality_audit = pd.DataFrame(columns=["reason", "final_class", "source", "n_rows"])

    t0 = time.time()
    raw_df = ingest.load_all_raw(cfg["paths"])
    stage_counts["raw_loaded"] = len(raw_df)
    timing["ingest_sec"] = round(time.time() - t0, 2)

    t0 = time.time()
    labeled_df, drop_summary = label_mod.apply_labels(raw_df)
    stage_counts["after_icd_labeling"] = len(labeled_df)
    labeled_df["anamnesa"] = labeled_df["anamnesa"].apply(clean.clean_anamnesa_text)

    # Safety check: label.py dirancang supaya HANYA pernah menghasilkan 12 kelas
    # di config['classes'] (Pneumonia/ISPA sudah digabung sejak label stage,
    # bukan langkah terpisah) -- ini menjaga taksonomi identik across v1/v2/v3
    # sesuai keputusan yang sudah dikunci di config/experiment.yaml.
    allowed_classes = set(cfg["classes"])
    unexpected = set(labeled_df["final_class"].unique()) - allowed_classes
    if unexpected:
        raise AssertionError(
            f"label.py menghasilkan kelas di luar daftar terkunci config: {unexpected}"
        )
    timing["label_sec"] = round(time.time() - t0, 2)

    t0 = time.time()
    if data_version in ("v2", "v3", "v4"):
        labeled_df = clean.remove_control_visits_without_complaint(labeled_df)
        labeled_df = clean.remove_incidental_covid(labeled_df)
    stage_counts["after_v2_cleaning"] = len(labeled_df)
    if data_version == "v4":
        labeled_df, quality_audit = clean.apply_v4_quality_filters(labeled_df)
        # Dedup membutuhkan ID asli, tetapi CSV/Kaggle dataset tidak. Hash ID
        # sebelum split/penulisan agar nomor RM mentah tidak keluar dari tahap build.
        labeled_df = clean.pseudonymize_record_ids(labeled_df)
    if data_version in ("v3", "v4"):
        labeled_df = clean.flag_unreliable_classes(
            labeled_df, min_support=cfg.get("min_support_reliable", 30)
        )
    stage_counts["final_before_split"] = len(labeled_df)
    timing["clean_sec"] = round(time.time() - t0, 2)

    return labeled_df, timing, drop_summary, quality_audit, stage_counts


def main(data_version: str, config_path: str = "config/experiment.yaml",
         output_dir: str | None = None) -> dict:
    cfg = split_mod.load_config(config_path)
    output_dir = output_dir or os.path.join(cfg["paths"]["processed_dir"], data_version)
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    labeled_df, timing, drop_summary, quality_audit, stage_counts = build_dataset(data_version, cfg)

    t0 = time.time()
    sp = cfg["split"]
    split_fn = split_mod.stratified_group_split if data_version == "v4" else split_mod.stratified_split
    train_df, val_df, test_df = split_fn(
        labeled_df, ratios=(sp["train"], sp["val"], sp["test"]), seed=cfg["seed"]
    )
    timing["split_sec"] = round(time.time() - t0, 2)
    timing["total_sec"] = round(time.time() - t_start, 2)

    # CSV (bukan parquet) -- sengaja, supaya tahap data ini tetap jalan di
    # environment mana pun tanpa pyarrow (mis. bridge tanpa akses internet
    # utk pip install). Ukuran data ini (~5-6rb baris) kecil, CSV cukup.
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    class_dist = {
        "train": train_df["final_class"].value_counts().to_dict(),
        "val": val_df["final_class"].value_counts().to_dict(),
        "test": test_df["final_class"].value_counts().to_dict(),
    }
    with open(os.path.join(output_dir, "class_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(class_dist, f, ensure_ascii=False, indent=2)

    drop_summary.to_csv(os.path.join(output_dir, "dropped_icd_summary.csv"), index=False)
    quality_audit.to_csv(os.path.join(output_dir, "quality_audit.csv"), index=False)

    summary = {
        "data_version": data_version,
        "n_total": len(labeled_df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_classes": labeled_df["final_class"].nunique(),
        "n_rows_written": len(train_df) + len(val_df) + len(test_df),
        "n_rows_lost_during_split": len(labeled_df) - (len(train_df) + len(val_df) + len(test_df)),
        "stage_counts": stage_counts,
        "timing_per_stage": timing,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Selesai build data-%s: %s", data_version, json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", choices=["v1", "v2", "v3", "v4"], required=True)
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    main(args.data_version, args.config)
