"""
Training entrypoint tunggal -- ganti v5_fixed.py / v6_training.py / v7_ensemble.py
yang sebelumnya jadi 3 file terpisah per iterasi.

Jalankan salah satu dari 6 kombinasi wajib (lihat rancangan-pipeline-redo.html):
    python -m src.train --data-version v1 --model base
    python -m src.train --data-version v1 --model large
    python -m src.train --data-version v2 --model base
    python -m src.train --data-version v2 --model large
    python -m src.train --data-version v3 --model base
    python -m src.train --data-version v3 --model large

BUTUH GPU -- jalankan di Kaggle (lihat README.md bagian "Cara run di Kaggle").
Di CPU (Mac/bridge biasa) ini akan jalan tapi sangat lambat untuk IndoBERT-large
(estimasi berjam-jam untuk dataset ~5000 baris x 12 epoch) -- jangan di-run
penuh di luar GPU kecuali cuma smoke-test (lihat --smoke-test).

Logika inti (focal loss, symptom flags, dropout-after-load fix) di-port dari
v5_fixed.py -- SATU perubahan: flag [CAMPAK] dihapus karena kelas Campak sudah
tidak ada di taksonomi manapun (v1/v2/v3), sesuai rekomendasi PLAN_V6_DATA_FIX.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

from src.evaluate import compute_metrics, build_hf_compute_metrics
from src.split import load_config
from src.pipeline import log_run

BULAN_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}


def extract_symptom_flags(text: str) -> str:
    """Port dari v5_fixed.py, TANPA flag [CAMPAK] (kelas Campak sudah dihapus total)."""
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    flags = []

    d = []
    if any(k in t for k in ["nyeri belakang mata", "nyeri retro", "retroorbital"]):
        d.append("nyeri retro-orbital")
    if any(k in t for k in ["petechiae", "petekie", "bintik merah", "bintik perdarahan", "rumple leed", "tourniquet"]):
        d.append("petechiae")
    if any(k in t for k in ["trombosit", "platelet", "trombositopeni"]):
        d.append("trombosit")
    if any(k in t for k in ["mimisan", "gusi berdarah", "epistaksis", "hidung berdarah"]):
        d.append("perdarahan mukosa")
    if any(k in t for k in ["nyeri sendi", "nyeri otot", "myalgia", "arthralgia", "pegal"]):
        d.append("nyeri sendi/otot")
    if d:
        flags.append(f"[DENGUE: {', '.join(d)}]")

    cv = []
    if any(k in t for k in ["anosmia", "tidak bisa mencium", "hilang penciuman", "ageusia"]):
        cv.append("anosmia")
    if any(k in t for k in ["kontak covid", "riwayat kontak", "kontak erat", "pcr positif", "antigen positif", "swab positif"]):
        cv.append("kontak covid")
    if any(k in t for k in ["saturasi", "spo2", "sesak nafas", "sesak napas"]):
        cv.append("respirasi/saturasi")
    if cv:
        flags.append(f"[COVID: {', '.join(cv)}]")

    p = []
    if any(k in t for k in ["napas cepat", "takipnea", "retraksi", "retraksi dinding dada"]):
        p.append("takipnea/retraksi")
    if any(k in t for k in ["ronkhi", "ronchi", "wheezing", "crackles"]):
        p.append("ronkhi/wheezing")
    if any(k in t for k in ["foto toraks", "rontgen", "x-ray", "infiltrat"]):
        p.append("foto toraks")
    if any(k in t for k in ["sianosis", "biru", "kebiruan"]):
        p.append("sianosis")
    if p:
        flags.append(f"[RESPIRASI BERAT: {', '.join(p)}]")

    return " ".join(flags)


def build_input(row) -> str:
    anamnesa = str(row.get("anamnesa", "") or "").strip()
    usia = row.get("age_years", None)
    usia_str = f"{int(usia)} tahun" if pd.notna(usia) else "tidak diketahui"
    sex = str(row.get("sex", "") or "").strip()
    if not sex or sex.lower() in ("nan", ""):
        sex = "tidak diketahui"
    bulan = row.get("bulan_kunjung", None)
    bulan_str = BULAN_ID.get(int(bulan), str(bulan)) if pd.notna(bulan) else "tidak diketahui"
    flags = extract_symptom_flags(anamnesa)
    prefix = f"{flags} " if flags else ""
    return f"{prefix}Usia: {usia_str}. Jenis kelamin: {sex}. Bulan: {bulan_str}. Anamnesa: {anamnesa}"


class PenyakitDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label_encoder):
        self.texts = [build_input(row) for _, row in dataframe.iterrows()]
        self.labels = label_encoder.transform(dataframe["final_class"].values)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len, padding="max_length",
                              truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, num_classes=12):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits, labels):
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_labels = torch.full_like(logits, self.label_smoothing / self.num_classes)
                smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes)
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_labels * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[labels] * focal_loss
        return focal_loss.mean()


def make_focal_trainer_class(focal_loss_fn):
    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = focal_loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss
    return FocalTrainer


def cleanup_old_checkpoints(output_dir: str) -> list[str]:
    """
    Hapus checkpoint-XXXX/ bikinan HF Trainer (save_total_limit=2 -> sampai 2
    salinan model penuh) di dalam output_dir. Dipanggil SETELAH
    trainer.save_model(output_dir) -- best model udah di-flatten ke root
    output_dir, jadi checkpoint-XXXX/ di dalamnya sudah redundan.

    Kenapa ini penting: di Kaggle /kaggle/working ruangnya terbatas (~20GB).
    6 kombinasi x sampai 2 checkpoint + 1 salinan final per kombinasi bisa
    numpuk sampai puluhan GB, bikin kombinasi terakhir (biasanya model large
    yg paling besar) gagal nyimpen dgn `OSError: No space left on device`
    walau kombinasi sebelumnya semua sukses -- itu yang kejadian di Kaggle.

    Non-fatal: kalau satu folder gagal dihapus (jarang, tapi bisa krn file
    lock dsb), lanjut ke folder lain, jangan gagalin training gara-gara ini.
    Return list path yang berhasil dihapus (buat testing/logging).
    """
    removed = []
    for ckpt_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        try:
            shutil.rmtree(ckpt_dir)
            removed.append(ckpt_dir)
        except OSError as e:
            print(f"[train] gagal hapus checkpoint lama '{ckpt_dir}' (dilanjut, bukan fatal): {e}")
    return removed


def run(data_version: str, model_key: str, cfg_path: str = "config/experiment.yaml",
        smoke_test: bool = False) -> dict:
    cfg = load_config(cfg_path)
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(cfg["paths"]["processed_dir"], data_version)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    if smoke_test:
        train_df = train_df.groupby("final_class", group_keys=False).head(2)
        val_df = val_df.groupby("final_class", group_keys=False).head(2)
        test_df = test_df.groupby("final_class", group_keys=False).head(2)

    reliable_mask = None
    if "is_reliable_class" in train_df.columns:
        reliable_mask = (
            train_df.groupby("final_class")["is_reliable_class"].first().to_dict()
        )

    model_name = cfg["models"][model_key]["hf_name"]
    run_id = f"{data_version}_{model_key}"
    output_dir = os.path.join("experiments", run_id)
    os.makedirs(output_dir, exist_ok=True)

    le = LabelEncoder()
    le.fit(sorted(set(train_df["final_class"]) | set(val_df["final_class"]) | set(test_df["final_class"])))
    num_classes = len(le.classes_)

    hp = cfg["training"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = PenyakitDataset(train_df, tokenizer, hp["max_len"], le)
    val_ds = PenyakitDataset(val_df, tokenizer, hp["max_len"], le)
    test_ds = PenyakitDataset(test_df, tokenizer, hp["max_len"], le)

    train_labels_arr = le.transform(train_df["final_class"].values)
    class_counts = np.bincount(train_labels_arr, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(class_counts + 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    focal_loss_fn = FocalLossWithSmoothing(
        gamma=hp["focal_gamma"], alpha=class_weights_tensor,
        label_smoothing=hp["label_smoothing"], num_classes=num_classes,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model.config.hidden_dropout_prob = hp["hidden_dropout"]
    model.config.classifier_dropout = hp["classifier_dropout"]
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if smoke_test else hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"] * 2,
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["lr"],
        weight_decay=hp["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="val_f1_macro_reliable_only" if reliable_mask else "val_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    FocalTrainer = make_focal_trainer_class(focal_loss_fn)
    trainer = FocalTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=build_hf_compute_metrics(list(le.classes_), reliable_mask, split_name="val"),
        callbacks=[] if smoke_test else [EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"])],
    )

    t0 = time.time()
    trainer.train()
    training_time_sec = round(time.time() - t0, 2)

    # load_best_model_at_end=True -> trainer.model sudah otomatis di-restore
    # ke checkpoint TERBAIK (bukan cuma epoch terakhir) begitu train() selesai.
    # Simpan model+tokenizer FINAL langsung di root output_dir (bukan cuma di
    # subfolder checkpoint-XXXX bikinan Trainer) supaya src/push_to_hf.py bisa
    # push langsung dari sini tanpa harus tebak-tebak checkpoint step mana yang
    # dipakai.
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    cleanup_old_checkpoints(output_dir)

    val_metrics = trainer.evaluate(val_ds)
    val_metrics = {k.replace("eval_", ""): v for k, v in val_metrics.items() if k.startswith("eval_val_")}

    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_metrics = compute_metrics(test_output.label_ids, test_preds, list(le.classes_), reliable_mask, split_name="test")

    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    record = {
        "run_id": run_id,
        "data_version": data_version,
        "model_key": model_key,
        "model_name": model_name,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "num_classes": num_classes,
        "training_time_sec": training_time_sec,
        "smoke_test": smoke_test,
        **val_metrics,
        **test_metrics,
    }
    log_run(record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--model", choices=["base", "large"], required=True)
    parser.add_argument("--smoke-test", action="store_true",
                         help="1 epoch, subset kecil per kelas -- utk cek pipeline jalan tanpa nunggu training penuh")
    args = parser.parse_args()
    result = run(args.data_version, args.model, smoke_test=args.smoke_test)
    print(json.dumps(result, ensure_ascii=False, indent=2))
