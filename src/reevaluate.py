"""
Re-evaluate model yang SUDAH dilatih (TANPA training ulang) pakai
`compute_metrics` yang udah di-fix -- bug lama (lihat src/evaluate.py) bikin
test_f1_macro / test_precision_macro / test_recall_macro / *_reliable_only
di SEMUA 6 run asli jadi 0.0, padahal test_accuracy normal (52%-72%).
Modelnya sendiri VALID (accuracy-nya bukti dia belajar beneran), yang salah
cuma cara hitung metriknya -- jadi solusinya cuma re-run evaluasi pakai
model+data yang sama, BUKAN retrain dari nol dari GPU lagi.

Prasyarat: folder `experiments/<run_id>/` masih ada isinya (config.json,
model.safetensors, tokenizer*, label_map.json) -- kalau sesi Kaggle-nya udah
mati dan folder ini udah ke-reset/ke-hapus (mis. abis dipakai
--push-hf-cleanup-local), TIDAK BISA dipakai lagi, harus training ulang.

Jalankan di sesi Kaggle yang SAMA (working dir yang masih ada model-nya),
dari root repo (biasanya /kaggle/working/repo):
    python -m src.reevaluate --run-id v1_base
    python -m src.reevaluate --all      # semua run_id yang foldernya masih ada

Setelah ini record baru (metrik yang sudah benar) ke-APPEND ke
experiments/runs.jsonl -- bukan overwrite baris lama. Ini aman karena
generate_report.py/push_to_hf.py sudah dirancang dedup by run_id, entri
PALING TERAKHIR yang menang -- jadi begitu script ini jalan, entri lama yang
metriknya 0.0 otomatis kalah/nggak kepakai lagi tanpa perlu edit manual.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate import compute_metrics
from src.pipeline import log_run
from src.push_to_hf import load_runs
from src.split import load_config
from src.train import PenyakitDataset


def class_names_from_label_map(label_map: dict) -> list[str]:
    """
    label_map.json isinya {"0": "NamaKelasA", "1": "NamaKelasB", ...} (lihat
    train.py::run -- label_map = {i: cls for i, cls in enumerate(le.classes_)}).
    Balikin list nama kelas terurut sesuai index integer-nya, supaya bisa
    dipakai identik dengan `le.classes_` aslinya waktu training.
    """
    return [label_map[str(i)] for i in range(len(label_map))]


def merge_record_with_new_metrics(old_record: dict, run_id: str,
                                   val_metrics: dict, test_metrics: dict) -> dict:
    """
    Gabungkan record LAMA (field non-metrik: model_name, n_train, dst -- kalau
    ada) dengan metrik val_/test_ yang BARU dihitung ulang. `old_record` boleh
    dict kosong kalau run_id-nya nggak ketemu di runs.jsonl (mis. dihapus
    manual) -- tetap jalan, cuma field non-metriknya nggak lengkap.
    """
    return {
        **old_record,
        "run_id": run_id,
        "reevaluated": True,
        **val_metrics,
        **test_metrics,
    }


def _predict(model, dataset, device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def reevaluate_run(run_id: str, cfg_path: str = "config/experiment.yaml",
                    experiments_dir: str = "experiments") -> dict:
    output_dir = os.path.join(experiments_dir, run_id)
    if not os.path.isdir(output_dir):
        raise RuntimeError(
            f"'{output_dir}' tidak ketemu -- model run '{run_id}' nggak ada di sesi ini, "
            f"kemungkinan sudah ke-cleanup atau sesi Kaggle-nya udah reset. Harus training ulang."
        )

    label_map_path = os.path.join(output_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        raise RuntimeError(f"'{label_map_path}' tidak ketemu -- nggak bisa rekonstruksi urutan kelas.")
    with open(label_map_path, "r", encoding="utf-8") as f:
        class_names = class_names_from_label_map(json.load(f))

    data_version = run_id.split("_")[0]
    cfg = load_config(cfg_path)
    data_dir = os.path.join(cfg["paths"]["processed_dir"], data_version)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))  # cuma dibutuhkan buat reliable_mask
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    reliable_mask = None
    if "is_reliable_class" in train_df.columns:
        reliable_mask = train_df.groupby("final_class")["is_reliable_class"].first().to_dict()

    le = LabelEncoder()
    le.classes_ = np.array(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

    hp = cfg["training"]
    eval_batch_size = hp["batch_size"] * 2
    val_ds = PenyakitDataset(val_df, tokenizer, hp["max_len"], le)
    test_ds = PenyakitDataset(test_df, tokenizer, hp["max_len"], le)

    val_labels, val_preds = _predict(model, val_ds, device, eval_batch_size)
    test_labels, test_preds = _predict(model, test_ds, device, eval_batch_size)

    val_metrics = compute_metrics(val_labels, val_preds, class_names, reliable_mask, split_name="val")
    test_metrics = compute_metrics(test_labels, test_preds, class_names, reliable_mask, split_name="test")

    existing_by_id = {r["run_id"]: r for r in load_runs() if not r.get("smoke_test")}
    old_record = existing_by_id.get(run_id, {})
    record = merge_record_with_new_metrics(old_record, run_id, val_metrics, test_metrics)
    log_run(record)
    return record


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Contoh: v1_base")
    group.add_argument("--all", action="store_true",
                        help="Re-evaluate semua run_id yang folder modelnya masih ada di experiments/")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()

    if args.all:
        run_ids = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(args.experiments_dir, "*"))
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "label_map.json"))
        )
        if not run_ids:
            raise SystemExit(f"Nggak ada model run_id yang ketemu di '{args.experiments_dir}'.")
    else:
        run_ids = [args.run_id]

    for run_id in run_ids:
        print(f"[reevaluate] '{run_id}' ...")
        try:
            record = reevaluate_run(run_id, cfg_path=args.config, experiments_dir=args.experiments_dir)
            print(f"    OK -- test_accuracy={record.get('test_accuracy')} "
                  f"test_f1_macro={record.get('test_f1_macro')} "
                  f"test_f1_macro_reliable_only={record.get('test_f1_macro_reliable_only')}")
        except RuntimeError as e:
            print(f"    GAGAL -- {e}")


if __name__ == "__main__":
    main()
