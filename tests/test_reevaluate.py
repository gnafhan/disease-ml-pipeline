"""
Test bagian logic murni src/reevaluate.py (parsing label_map.json, gabung
record lama+baru). Nggak nge-test bagian load model/tokenizer asli (butuh
GPU/internet buat download IndoBERT) -- itu cuma bisa diverifikasi langsung
di Kaggle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reevaluate import class_names_from_label_map, merge_record_with_new_metrics


def test_class_names_from_label_map_preserves_index_order():
    label_map = {"0": "Diare Akut", "1": "Suspek Dengue", "2": "Pneumonia/ISPA"}
    assert class_names_from_label_map(label_map) == ["Diare Akut", "Suspek Dengue", "Pneumonia/ISPA"]


def test_class_names_from_label_map_handles_unsorted_keys():
    # dict JSON bisa aja urutan key-nya nggak berurutan pas dibaca -- fungsi
    # HARUS urut berdasarkan nilai integer key-nya, bukan urutan insersi dict.
    label_map = {"2": "C", "0": "A", "1": "B"}
    assert class_names_from_label_map(label_map) == ["A", "B", "C"]


def test_merge_record_keeps_old_non_metric_fields():
    old_record = {
        "run_id": "v1_base", "model_name": "indolem/indobert-base-uncased",
        "n_train": 4079, "smoke_test": False,
        "test_f1_macro": 0.0, "test_accuracy": 0.6,  # metrik lama yang salah
    }
    val_metrics = {"val_accuracy": 0.7, "val_f1_macro": 0.55}
    test_metrics = {"test_accuracy": 0.6, "test_f1_macro": 0.52}  # metrik baru yang benar

    merged = merge_record_with_new_metrics(old_record, "v1_base", val_metrics, test_metrics)

    assert merged["model_name"] == "indolem/indobert-base-uncased"  # field lama tetap ada
    assert merged["n_train"] == 4079
    assert merged["test_f1_macro"] == 0.52  # metrik lama ke-overwrite yang baru
    assert merged["reevaluated"] is True


def test_merge_record_works_even_without_old_record():
    # run_id yang nggak ketemu di runs.jsonl lama (mis. dihapus manual) --
    # tetap harus jalan, bukan crash.
    merged = merge_record_with_new_metrics({}, "v3_large", {"val_f1_macro": 0.5}, {"test_f1_macro": 0.6})
    assert merged["run_id"] == "v3_large"
    assert merged["test_f1_macro"] == 0.6
