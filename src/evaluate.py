"""
Evaluate -- Tahap 07. TERKUNCI. SATU fungsi metrik dipakai untuk SEMUA run.

Ini yang mencegah kejadian kemarin: "V6 final" headline pakai Val Macro F1
(70.1%) padahal versi lain di tabel pakai Test Macro F1 -- dua metrik beda
yang keliatan dibandingkan langsung.

ATURAN: metrik final yang dilaporkan SELALU dihitung di TEST SET.
Val-set metrics (kalau dicatat) WAJIB pakai prefix `val_` di key-nya,
supaya tidak pernah tertukar jadi angka headline.
"""
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


def compute_metrics(y_true, y_pred, classes: list, min_support_reliable: int) -> dict:
    """Return dict siap ditulis ke experiments/runs.jsonl:
        accuracy, weighted_f1,
        macro_f1_all              (semua kelas, termasuk yang support kecil)
        macro_f1_reliable_only    (cuma kelas dengan support >= min_support_reliable)
        per_class: {kelas: {precision, recall, f1, support}}
        confusion_top_pairs: [(true, pred, count), ...] -- top 5 error pairs

    TODO: implementasi. Pastikan macro_f1_reliable_only dihitung dari SUBSET
    kelas yang reliable (bukan classification_report biasa yang otomatis
    rata-rata semua kelas).
    """
    raise NotImplementedError
