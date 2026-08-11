"""
Load config terkunci & stratified split 70/15/15 dengan leakage check.

SENGAJA tidak bergantung ke scikit-learn -- tahap data (ingest/label/clean/split)
ini harus bisa jalan di environment CPU-only mana pun (termasuk lewat bridge
tanpa akses internet utk pip install), jadi stratified split diimplementasi
manual pakai pandas/numpy saja. scikit-learn baru dibutuhkan di evaluate.py,
yang HANYA jalan di tahap training (Kaggle, sudah pre-installed di sana).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/experiment.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _manual_stratified_split(df: pd.DataFrame, class_col: str, test_size: float, seed: int):
    """
    Ganti sklearn.train_test_split(..., stratify=...): untuk tiap kelas, ambil
    proporsi test_size secara acak (seed tetap), sisanya ke train. Kelas dengan
    cuma 1 anggota otomatis seluruhnya ke train (round ke 0 baris test) -- jadi
    tidak perlu penanganan khusus/error seperti versi sklearn.
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    too_rare = []
    for cls, group in df.groupby(class_col, sort=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        if len(idx) < 2:
            too_rare.append(cls)
            n_test = 0
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    if too_rare:
        logger.warning(
            "%d kelas cuma punya 1 sampel total, semua barisnya dipaksa ke 'train': %s",
            len(too_rare), too_rare,
        )
    train_df = df.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df


def _leak_mask(target_df: pd.DataFrame, ref_df: pd.DataFrame,
                record_id_col: str, text_col: str) -> pd.Series:
    """
    Sinyal utama leakage = record_id (No.RM / hash Patient) yang sudah muncul
    di ref_df -- ini akurat karena record_id unik per pasien/kunjungan.

    Match teks anamnesa PERSIS cuma dipakai sbg fallback utk baris yang
    record_id-nya kosong/NaN. TIDAK dipakai sbg sinyal utama (beda dari
    v5_fixed.py lama) karena banyak kalimat generik pendek ("kontrol tdk ada
    keluhan") bisa persis sama antar pasien BEDA yang tidak benar-benar
    duplikat -- kalau dipakai sbg sinyal utama, val/test bisa nyaris kosong
    (ketemu saat uji dgn data sintetis di sesi ini).
    """
    ref_ids = set(ref_df[record_id_col].dropna().astype(str))
    ref_ids.discard("")
    ref_ids.discard("nan")

    id_leak = target_df[record_id_col].astype(str).isin(ref_ids)

    has_no_id = target_df[record_id_col].isna() | (target_df[record_id_col].astype(str).str.strip().isin(["", "nan"]))
    ref_texts = set(ref_df[text_col].dropna().astype(str).str.strip())
    text_leak = has_no_id & target_df[text_col].astype(str).str.strip().isin(ref_texts)

    return id_leak | text_leak


def stratified_split(df: pd.DataFrame, class_col: str = "final_class",
                      ratios=(0.70, 0.15, 0.15), seed: int = 42,
                      record_id_col: str = "record_id", text_col: str = "anamnesa",
                      max_leak_drop_ratio: float = 0.4):
    """
    Split 70/15/15 stratified per kelas (patient-level leak guard): baris
    val/test yang record_id-nya sudah muncul di train/val sebelumnya dibuang
    dari val/test (bukan dari train) -- lihat _leak_mask utk kenapa record_id
    dipilih sbg sinyal utama, bukan match teks.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios harus total 1.0"
    train_ratio, val_ratio, test_ratio = ratios

    train_df, rest_df = _manual_stratified_split(df, class_col, test_size=(1 - train_ratio), seed=seed)

    val_frac_of_rest = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = _manual_stratified_split(rest_df, class_col, test_size=(1 - val_frac_of_rest), seed=seed)

    def _drop_leak(target_df, ref_df, name):
        leak_mask = _leak_mask(target_df, ref_df, record_id_col, text_col)
        n_leak = int(leak_mask.sum())
        if n_leak:
            drop_ratio = n_leak / max(len(target_df), 1)
            log_fn = logger.warning if drop_ratio > max_leak_drop_ratio else logger.info
            log_fn("stratified_split: %d/%d (%.0f%%) baris %s dibuang (leakage) -- "
                   "%s", n_leak, len(target_df), drop_ratio * 100, name,
                   "TERLALU BANYAK, cek data (kemungkinan banyak record_id duplikat)"
                   if drop_ratio > max_leak_drop_ratio else "wajar")
        return target_df[~leak_mask].reset_index(drop=True)

    val_df = _drop_leak(val_df, train_df, "val")
    test_df = _drop_leak(test_df, train_df, "test")
    test_df = _drop_leak(test_df, val_df, "test (vs val)")

    logger.info("stratified_split final: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df, test_df
