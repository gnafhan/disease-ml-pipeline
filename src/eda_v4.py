"""Generate engineering EDA and V4 decision report (aggregate-only).

The report deliberately never writes anamnesis text, record IDs, or other
patient-level values. It compares V3 and V4, runs cheap CPU baselines, measures
leakage/noise, and records the evidence behind every V4 transform.

Usage:
    python -m src.eda_v4
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.clean import V4_CLASS_ANCHOR_PATTERNS, canonicalize_anamnesa, has_v4_class_anchor


SPLITS = ("train", "val", "test")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def load_version(version: str, processed_dir: str = "data/processed") -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    parts = {split: pd.read_csv(os.path.join(processed_dir, version, f"{split}.csv")) for split in SPLITS}
    all_rows = pd.concat([frame.assign(_split=split) for split, frame in parts.items()], ignore_index=True)
    return parts, all_rows


def _summary(version: str, parts: dict[str, pd.DataFrame], all_rows: pd.DataFrame,
             processed_dir: str) -> dict:
    build_path = os.path.join(processed_dir, version, "build_summary.json")
    build = json.load(open(build_path, encoding="utf-8")) if os.path.exists(build_path) else {}
    canonical = all_rows["anamnesa"].apply(canonicalize_anamnesa)
    template_sizes = canonical.value_counts()
    conflicts = all_rows.assign(_canonical=canonical).groupby("_canonical")["final_class"].nunique()

    patient_sets = {split: set(frame["record_id"].astype(str)) for split, frame in parts.items()}
    text_sets = {
        split: set(frame["anamnesa"].apply(canonicalize_anamnesa)) - {""}
        for split, frame in parts.items()
    }
    patient_overlap = sum(len(patient_sets[a] & patient_sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")])
    text_overlap = sum(len(text_sets[a] & text_sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")])

    record_class_counts = all_rows.groupby(all_rows["record_id"].astype(str))["final_class"].nunique()
    multi_class_ids = set(record_class_counts[record_class_counts > 1].index)
    word_count = canonical.str.split().str.len().fillna(0)
    return {
        "version": version,
        "rows_before_split": int(build.get("n_total", len(all_rows))),
        "rows_written": int(len(all_rows)),
        "rows_lost_during_split": int(build.get("n_total", len(all_rows)) - len(all_rows)),
        "split_rows": {split: int(len(frame)) for split, frame in parts.items()},
        "n_classes": int(all_rows["final_class"].nunique()),
        "stage_counts": build.get("stage_counts", {}),
        "empty_rows": int(canonical.eq("").sum()),
        "rows_in_duplicate_templates": int(canonical.isin(template_sizes[template_sizes > 1].index).sum()),
        "duplicate_template_groups": int((template_sizes > 1).sum()),
        "conflicting_template_groups": int((conflicts > 1).sum()),
        "conflicting_template_rows": int(canonical.isin(conflicts[conflicts > 1].index).sum()),
        "multi_class_patient_ids": int(len(multi_class_ids)),
        "rows_from_multi_class_patients": int(all_rows["record_id"].astype(str).isin(multi_class_ids).sum()),
        "patient_overlap_groups": int(patient_overlap),
        "template_overlap_groups": int(text_overlap),
        "median_words": float(word_count.median()),
        "p95_words": float(word_count.quantile(0.95)),
    }


def _class_profile(all_rows: pd.DataFrame) -> list[dict]:
    rows = []
    total_by_class = all_rows["final_class"].value_counts()
    source_counts = all_rows.groupby(["final_class", "source"]).size()
    for class_name, count in total_by_class.items():
        sources = source_counts.loc[class_name]
        max_source = str(sources.idxmax())
        max_share = float(sources.max() / count)
        anchor_count = sum(
            has_v4_class_anchor(text, class_name)
            for text in all_rows.loc[all_rows["final_class"] == class_name, "anamnesa"].fillna("")
        )
        rows.append({
            "class": class_name,
            "rows": int(count),
            "largest_source": max_source,
            "largest_source_share": round(max_share, 4),
            "anchor_coverage": round(anchor_count / count, 4),
            "reliable": bool(count >= 30),
        })
    return rows


def _metric_bundle(y_true, y_pred, classes: list[str], reliable: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro_all": round(float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)), 4),
        "f1_macro_reliable": round(float(f1_score(y_true, y_pred, labels=reliable, average="macro", zero_division=0)), 4),
    }


def _text_baseline(parts: dict[str, pd.DataFrame]) -> dict:
    train, test = parts["train"], parts["test"]
    classes = sorted(train["final_class"].unique())
    reliable = sorted(
        train.groupby("final_class")["is_reliable_class"].first().loc[lambda values: values].index
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True),
        LinearSVC(class_weight="balanced"),
    )
    model.fit(train["anamnesa"].fillna(""), train["final_class"])
    predictions = model.predict(test["anamnesa"].fillna(""))
    result = _metric_bundle(test["final_class"], predictions, classes, reliable)

    matrix = confusion_matrix(test["final_class"], predictions, labels=classes)
    confusions = []
    for true_idx, true_class in enumerate(classes):
        for pred_idx, pred_class in enumerate(classes):
            if true_idx != pred_idx and matrix[true_idx, pred_idx]:
                confusions.append({
                    "count": int(matrix[true_idx, pred_idx]),
                    "true": true_class,
                    "predicted": pred_class,
                })
    result["top_confusions"] = sorted(confusions, key=lambda row: -row["count"])[:12]
    return result


def _metadata_baselines(parts: dict[str, pd.DataFrame]) -> dict:
    train, test = parts["train"], parts["test"]
    classes = sorted(train["final_class"].unique())
    reliable = sorted(
        train.groupby("final_class")["is_reliable_class"].first().loc[lambda values: values].index
    )
    variants = {
        "clinical_metadata_only": ["age_years", "sex", "bulan_kunjung"],
        "source_style_only": ["source", "visit_type"],
        "all_metadata": ["age_years", "sex", "bulan_kunjung", "source", "visit_type"],
    }
    output = {}
    for name, columns in variants.items():
        numeric = [column for column in columns if column in {"age_years", "bulan_kunjung"}]
        categorical = [column for column in columns if column not in numeric]
        transformers = []
        if numeric:
            transformers.append(("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
        if categorical:
            transformers.append(("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
            ), categorical))
        model = make_pipeline(
            ColumnTransformer(transformers),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        model.fit(train[columns], train["final_class"])
        predictions = model.predict(test[columns])
        output[name] = _metric_bundle(test["final_class"], predictions, classes, reliable)
    return output


def build_metrics(processed_dir: str = "data/processed") -> dict:
    metrics = {"versions": {}, "method": {"raw_text_in_report": False}}
    for version in ("v3", "v4"):
        parts, all_rows = load_version(version, processed_dir)
        metrics["versions"][version] = {
            "summary": _summary(version, parts, all_rows, processed_dir),
            "class_profile": _class_profile(all_rows),
            "text_baseline": _text_baseline(parts),
            "metadata_baselines": _metadata_baselines(parts),
        }

    audit_path = os.path.join(processed_dir, "v4", "quality_audit.csv")
    audit = pd.read_csv(audit_path)
    metrics["v4_quality_actions"] = {
        reason: int(count)
        for reason, count in audit.groupby("reason")["n_rows"].sum().items()
    }
    return _json_safe(metrics)


def render_report(metrics: dict) -> str:
    v3 = metrics["versions"]["v3"]
    v4 = metrics["versions"]["v4"]
    s3, s4 = v3["summary"], v4["summary"]
    actions = metrics["v4_quality_actions"]
    removed = sum(count for reason, count in actions.items() if reason != "kept")

    class_rows = []
    profile4 = {row["class"]: row for row in v4["class_profile"]}
    for row in sorted(v3["class_profile"], key=lambda item: -item["rows"]):
        newer = profile4[row["class"]]
        class_rows.append([
            row["class"], row["rows"], newer["rows"],
            f"{newer['largest_source_share'] * 100:.1f}% {newer['largest_source']}",
            f"{newer['anchor_coverage'] * 100:.1f}%",
            "yes" if newer["reliable"] else "no",
        ])

    baseline_rows = []
    for version, data in [("V3", v3), ("V4", v4)]:
        b = data["text_baseline"]
        baseline_rows.append([version, b["accuracy"], b["f1_macro_all"], b["f1_macro_reliable"]])

    confusion_rows = [
        [row["true"], row["predicted"], row["count"]]
        for row in v4["text_baseline"]["top_confusions"][:10]
    ]
    action_rows = [[reason, count] for reason, count in sorted(actions.items())]

    metadata3 = v3["metadata_baselines"]
    metadata4 = v4["metadata_baselines"]

    return f"""# Laporan Engineering EDA & Keputusan Data V4

Di-generate oleh `python -m src.eda_v4`. Report ini hanya memuat statistik agregat; teks anamnesis dan identifier pasien tidak pernah ditulis ke sini.

## Hasil

V4 mempertahankan taksonomi 12 kelas, tetapi mengubah quality gate dan cara split. Pipeline menulis {s4['rows_written']:,} baris ({s4['split_rows']['train']:,}/{s4['split_rows']['val']:,}/{s4['split_rows']['test']:,}) dan tidak kehilangan baris saat split. V3 menghasilkan 5.249 baris bersih tetapi hanya menulis {s3['rows_written']:,}; {s3['rows_lost_during_split']:,} baris ({s3['rows_lost_during_split']/s3['rows_before_split']:.1%}) dibuang oleh leak guard setelah split.

Baseline teks CPU naik dari macro F1 kelas reliable {v3['text_baseline']['f1_macro_reliable']:.4f} menjadi {v4['text_baseline']['f1_macro_reliable']:.4f}. Ini engineering gate, bukan perbandingan model final: V4 memakai test split patient/template-grouped yang lebih ketat dan test set yang lebih besar.

## Failure mode yang ditemukan di V3

- **Baris hilang setelah split:** {s3['rows_lost_during_split']:,}. Algoritma lama membagi baris lebih dulu, lalu menghapus kunjungan val/test bila pasiennya muncul di subset sebelumnya. V4 menetapkan connected group pasien/template sebelum split.
- **Header Excel bergeser:** delapan sheet RSUD menaruh header asli pada baris Excel 2–7. Loader lama melewatkan 25 baris pasien. Deteksi header adaptif menaikkan raw ingestion dari 5.908 menjadi {s4['stage_counts'].get('raw_loaded', 5933):,} baris.
- **Anamnesis kosong:** {s3['empty_rows']:,} baris tersimpan tanpa sinyal teks.
- **Duplikasi template:** {s3['rows_in_duplicate_templates']:,} baris masuk ke {s3['duplicate_template_groups']:,} template ternormalisasi yang berulang.
- **Teks identik dengan label konflik:** {s3['conflicting_template_rows']:,} baris dalam {s3['conflicting_template_groups']:,} group template mempunyai lebih dari satu label. Mayoritas berupa catatan kosong; satu template kontrol post-ranap generik juga melintasi label.
- **Pasien multi-kondisi:** {s3['rows_from_multi_class_patients']:,} baris dari {s3['multi_class_patient_ids']:,} ID pasien mencakup lebih dari satu kelas. Baris ini tidak otomatis salah, tetapi wajib berada dalam subset yang sama.
- **Konsentrasi sumber:** beberapa kelas hampir seluruhnya berasal dari satu rumah sakit. Random internal split mengukur in-domain recognition, bukan generalisasi lintas rumah sakit.
- **Evaluasi kelas langka:** lima kelas V4 tetap mempunyai kurang dari 30 baris. Test support-nya hanya 1–4 kasus per kelas, sehingga satu prediksi bisa mengubah F1 secara ekstrem.

## Tindakan quality gate V4

{_markdown_table(['action', 'rows'], action_rows)}

Total baris yang dibuang aturan khusus V4: {removed:,}. Gate catatan kontrol/sangat pendek memakai sinyal klinis label-agnostic. Kecocokan anchor terhadap kelas hanya dihitung sebagai diagnostik; nilainya tidak pernah menentukan kept/drop atau mengubah label.

## Profil kelas, sumber, dan sinyal teks

{_markdown_table(['kelas', 'baris V3', 'baris V4', 'sumber terbesar V4', 'cakupan anchor kelas', '>=30 baris'], class_rows)}

Source share mendekati 100% adalah deployment risk: kelas dan gaya dokumentasi saling terikat. V4 mempertahankan baris tersebut, tetapi melakukan stratifikasi `class x source` sambil mengelompokkan pasien/template. Pengukuran performa eksternal tetap membutuhkan rumah sakit lain atau periode waktu yang lebih baru.

## Baseline CPU murah

Baseline memakai TF-IDF word 1–2 gram dan class-balanced LinearSVC. Ini bukan model produksi; fungsinya menangkap regresi pipeline sebelum memakai kuota GPU.

{_markdown_table(['dataset', 'accuracy', 'macro F1 semua 12', 'macro F1 reliable 7'], baseline_rows)}

Arah confusion terbesar V4:

{_markdown_table(['kelas aktual', 'kelas prediksi', 'baris'], confusion_rows)}

Overlap dominan tetap berada pada COVID-19, Pneumonia/ISPA, Diare Akut, dan Suspek Dengue. Catatan mereka sama-sama memuat demam, batuk, mual/muntah, dan diare. Cleaning dapat membuang noise yang jelas, tetapi tidak dapat menciptakan bukti klinis diskriminatif yang memang tidak pernah dicatat.

## Diagnostik metadata/sumber

Usia/sex/bulan saja menghasilkan reliable macro F1 {metadata3['clinical_metadata_only']['f1_macro_reliable']:.4f} pada V3 dan {metadata4['clinical_metadata_only']['f1_macro_reliable']:.4f} pada V4. Source/visit type saja menghasilkan {metadata3['source_style_only']['f1_macro_reliable']:.4f} dan {metadata4['source_style_only']['f1_macro_reliable']:.4f}. Accuracy source-only V3 sebesar {metadata3['source_style_only']['accuracy']:.4f}, terutama karena COVID-19 hanya ada di RS Akademik UGM; pada split V4 turun menjadi {metadata4['source_style_only']['accuracy']:.4f}. Kolom `source` sengaja tidak diberikan ke neural model, tetapi gaya teks spesifik rumah sakit masih dapat membuka shortcut yang sama.

## Invariant V4

{_markdown_table(['pemeriksaan', 'V3', 'target/hasil V4'], [
    ['rows lost during split', s3['rows_lost_during_split'], s4['rows_lost_during_split']],
    ['patient overlap groups', s3['patient_overlap_groups'], s4['patient_overlap_groups']],
    ['template overlap groups', s3['template_overlap_groups'], s4['template_overlap_groups']],
    ['empty anamnesis', s3['empty_rows'], s4['empty_rows']],
    ['conflicting template groups', s3['conflicting_template_groups'], s4['conflicting_template_groups']],
    ['class count', s3['n_classes'], s4['n_classes']],
])}

## Keputusan yang sengaja tidak diambil

- V4 tidak menghapus kelas langka untuk menaikkan headline metric.
- V4 tidak me-relabel kelas berbasis ICD menggunakan keyword.
- V4 tidak melakukan oversampling dengan menyalin baris ke validation/test.
- V4 tidak mengklaim validasi lintas rumah sakit dari random internal split.
- V4 tidak memasukkan contoh teks mentah ke report karena sumber berisi data pasien.
- V4 menulis `record_id` sebagai SHA-256 pseudonym, bukan nomor RM sumber; dataset tetap diperlakukan sensitif dan private.

## Artefak audit setiap training

Selain `runs.jsonl` dan model, setiap run sekarang menulis `error_analysis.json` berisi confusion matrix, confusion pair terbesar, serta error rate agregat menurut source, visit type, kecocokan anchor, dan panjang teks. `run_manifest.json` merekam SHA-256 train/val/test dan config, git SHA, versi Python/library, device/GPU, checkpoint terbaik, global step, dan epoch selesai. Keduanya tidak menyimpan teks anamnesis atau ID pasien.

Dependency di `requirements.txt` belum dikunci sampai patch version, sehingga manifest runtime wajib disimpan bersama hasil. Untuk klaim final, bandingkan hash manifest dan pastikan load-best-checkpoint menghasilkan metrik yang sama ketika model di-reload.

## Gate eksperimen GPU

1. Jalankan smoke-test `v4_base` saja.
2. Jalankan full `v4_base`; persist `runs.jsonl`, metrik per kelas, confusion matrix, dan model sebelum kernel berhenti.
3. Bandingkan recall per kelas dan reliable-class macro F1 dengan V3, sambil mencatat bahwa split berubah.
4. Jalankan `v4_large` hanya jika V4 base stabil dan expected gain layak untuk sekitar 40–60 menit GPU.
5. Perlakukan lima kelas di bawah 30 baris sebagai insufficient-data classes terlepas dari test F1 yang volatil.
"""


def main(
    processed_dir: str = "data/processed",
    report_path: str = "reports/v4_engineering_eda.md",
    metrics_path: str = "reports/v4_eda_metrics.json",
) -> dict:
    metrics = build_metrics(processed_dir)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(metrics))
    print(f"Ditulis: {report_path}")
    print(f"Ditulis: {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()
