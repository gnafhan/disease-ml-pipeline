import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import compute_metrics


def test_compute_metrics_prefixes_keys_correctly():
    """Regression test langsung utk bug asli: key metrik HARUS diprefix val_/test_,
    tidak boleh ada key generik 'macro_f1' polos yang bisa ke-tuker val vs test."""
    y_true = ["A", "A", "B", "B", "C"]
    y_pred = ["A", "B", "B", "B", "C"]
    result = compute_metrics(y_true, y_pred, class_names=["A", "B", "C"], split_name="val")

    assert "val_f1_macro" in result
    assert "val_accuracy" in result
    assert "macro_f1" not in result  # key generik tanpa prefix TIDAK boleh ada
    assert "f1_macro" not in result

    result_test = compute_metrics(y_true, y_pred, class_names=["A", "B", "C"], split_name="test")
    assert "test_f1_macro" in result_test
    assert "val_f1_macro" not in result_test


def test_compute_metrics_reliable_only_matches_manual_calc():
    y_true = ["A", "A", "A", "B", "B", "C"]
    y_pred = ["A", "A", "B", "B", "B", "C"]
    reliable_mask = {"A": True, "B": True, "C": False}
    result = compute_metrics(y_true, y_pred, class_names=["A", "B", "C"],
                              reliable_mask_per_class=reliable_mask, split_name="test")
    assert "test_f1_macro_reliable_only" in result
    assert result["test_n_reliable_classes"] == 2
    # f1_macro_reliable_only cuma dihitung dari kelas A & B, harus beda dari f1_macro (semua kelas)
    assert result["test_f1_macro_reliable_only"] != result["test_f1_macro"]


def test_per_class_support_sums_to_total():
    y_true = ["A", "A", "B", "B", "B", "C"]
    y_pred = ["A", "B", "B", "A", "B", "C"]
    result = compute_metrics(y_true, y_pred, class_names=["A", "B", "C"], split_name="val")
    total_support = sum(v["support"] for v in result["val_per_class"].values())
    assert total_support == len(y_true)
