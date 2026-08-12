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


def test_compute_metrics_with_integer_labels_matches_string_labels():
    """
    Regression test utk bug nyata di data Kaggle: train.py SELALU manggil
    compute_metrics dengan y_true/y_pred berupa INDEX INTEGER (dari
    np.argmax + label_ids HF Trainer), bukan nama kelas string -- beda
    dengan test2 di atas yang (nggak realistis) manggil pakai string
    langsung. Sebelum fix, kombinasi integer labels + class_names string
    bikin precision_recall_fscore_support diam2 balik 0.0 semua walau
    accuracy-nya normal. Test ini WAJIB pakai integer index biar
    representatif sama pemanggilan asli di train.py.
    """
    class_names = ["A", "B", "C"]
    # index integer: A=0, B=1, C=2 (urutan sama kayak class_names, persis
    # kontrak `list(le.classes_)` di train.py)
    y_true_int = [0, 0, 1, 1, 1, 2]
    y_pred_int = [0, 1, 1, 0, 1, 2]
    y_true_str = ["A", "A", "B", "B", "B", "C"]
    y_pred_str = ["A", "B", "B", "A", "B", "C"]

    result_int = compute_metrics(y_true_int, y_pred_int, class_names=class_names, split_name="test")
    result_str = compute_metrics(y_true_str, y_pred_str, class_names=class_names, split_name="test")

    # kasus paling penting: metrik dari integer labels TIDAK BOLEH nol semua
    # kalau prediksinya memang ada yang benar (accuracy > 0 di sini).
    assert result_int["test_f1_macro"] > 0.0
    assert result_int["test_precision_macro"] > 0.0
    assert result_int["test_recall_macro"] > 0.0

    # dan hasilnya harus SAMA PERSIS antara versi integer vs versi string,
    # karena keduanya merepresentasikan data yang identik.
    assert result_int["test_accuracy"] == result_str["test_accuracy"]
    assert result_int["test_f1_macro"] == result_str["test_f1_macro"]
    assert result_int["test_precision_macro"] == result_str["test_precision_macro"]
    assert result_int["test_recall_macro"] == result_str["test_recall_macro"]


def test_compute_metrics_with_integer_labels_and_reliable_mask():
    class_names = ["A", "B", "C"]
    y_true_int = [0, 0, 0, 1, 1, 2]
    y_pred_int = [0, 0, 1, 1, 1, 2]
    reliable_mask = {"A": True, "B": True, "C": False}

    result = compute_metrics(y_true_int, y_pred_int, class_names=class_names,
                              reliable_mask_per_class=reliable_mask, split_name="test")

    assert result["test_f1_macro_reliable_only"] > 0.0
    assert result["test_n_reliable_classes"] == 2
