"""
Metrik evaluasi lengkap & konsisten -- dirancang khusus supaya TIDAK terulang
kebingungan val vs test yang terjadi di V6-Final (headline "Val Macro F1 70.1%"
padahal Test Macro F1 cuma 57.9%).

ATURAN WAJIB dipakai di train.py/pipeline.py:
  - Semua key metrik val HARUS diprefix 'val_' secara eksplisit.
  - Semua key metrik test HARUS diprefix 'test_' secara eksplisit.
  - JANGAN pernah nulis key metrik generik tanpa prefix ('macro_f1' polos) ke
    runs.jsonl -- itu sumber kebingungan sebelumnya. Prefix di-set di sini
    lewat parameter `split_name`, bukan diketik manual di banyak tempat.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
)


def compute_metrics(y_true, y_pred, class_names: list[str],
                     reliable_mask_per_class: dict[str, bool] | None = None,
                     split_name: str = "val") -> dict:
    """
    y_true, y_pred: array label index (int) ATAU nama kelas (str) -- keduanya
    harus konsisten & panjangnya sama dengan class_names kalau index.

    reliable_mask_per_class: {nama_kelas: True/False} dari clean.flag_unreliable_classes
    (dipakai untuk hitung f1_macro_reliable_only, metrik utama sesuai config.yaml).

    Return dict dengan key sudah diprefix `{split_name}_...` -- lihat docstring modul.
    """
    prefix = f"{split_name}_"

    accuracy = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="weighted", zero_division=0
    )
    p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None, zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(support_per[i]),
        }

    result = {
        f"{prefix}accuracy": round(float(accuracy), 4),
        f"{prefix}precision_macro": round(float(p_macro), 4),
        f"{prefix}recall_macro": round(float(r_macro), 4),
        f"{prefix}f1_macro": round(float(f1_macro), 4),
        f"{prefix}precision_weighted": round(float(p_weighted), 4),
        f"{prefix}recall_weighted": round(float(r_weighted), 4),
        f"{prefix}f1_weighted": round(float(f1_weighted), 4),
        f"{prefix}per_class": per_class,
    }

    if reliable_mask_per_class:
        reliable_classes = [c for c in class_names if reliable_mask_per_class.get(c, False)]
        if reliable_classes:
            _, _, f1_macro_reliable, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=reliable_classes, average="macro", zero_division=0
            )
            result[f"{prefix}f1_macro_reliable_only"] = round(float(f1_macro_reliable), 4)
            result[f"{prefix}n_reliable_classes"] = len(reliable_classes)

    return result


def build_hf_compute_metrics(class_names: list[str], reliable_mask_per_class: dict[str, bool] | None,
                              split_name: str = "val"):
    """
    Factory utk dipakai sbg `compute_metrics=` di transformers.Trainer.
    Trainer sendiri sudah nge-prefix 'eval_' otomatis saat logging -- fungsi ini
    TETAP nge-prefix val_/test_ secara eksplisit di dalam dict yg dikembalikan,
    supaya waktu run_result-nya diselamatkan ke runs.jsonl, key-nya sudah benar
    tanpa perlu rename manual (sumber bug sebelumnya).
    """
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds, class_names, reliable_mask_per_class, split_name=split_name)
    return _compute


def full_classification_report(y_true, y_pred, class_names: list[str]) -> str:
    return classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                  target_names=class_names, zero_division=0)
