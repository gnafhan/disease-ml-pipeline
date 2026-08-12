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
import re
import unicodedata

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _canonical_text_for_grouping(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"\d+(?:[.,]\d+)?", "<n>", text)
    text = re.sub(r"[^a-z<>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _connected_group_ids(
    df: pd.DataFrame,
    record_id_col: str,
    text_col: str,
) -> pd.Series:
    """Gabungkan baris yang berbagi pasien ATAU template teks yang sama.

    Union/transitive closure diperlukan karena satu pasien dapat memiliki
    beberapa teks, dan satu template dapat dipakai beberapa pasien. Seluruh
    komponen harus masuk subset yang sama agar tidak ada patient/template leak.
    """
    uf = _UnionFind(len(df))
    first_by_record: dict[str, int] = {}
    first_by_text: dict[str, int] = {}
    canonical_text = df[text_col].apply(_canonical_text_for_grouping)

    for position, (record_id, text) in enumerate(zip(df[record_id_col], canonical_text)):
        rid = "" if pd.isna(record_id) else str(record_id).strip()
        if rid and rid.lower() not in {"nan", "none"}:
            if rid in first_by_record:
                uf.union(position, first_by_record[rid])
            else:
                first_by_record[rid] = position
        if text:
            if text in first_by_text:
                uf.union(position, first_by_text[text])
            else:
                first_by_text[text] = position

    roots = [uf.find(i) for i in range(len(df))]
    root_to_group = {root: group_id for group_id, root in enumerate(sorted(set(roots)))}
    return pd.Series([root_to_group[root] for root in roots], index=df.index, dtype="int64")


def stratified_group_split(
    df: pd.DataFrame,
    class_col: str = "final_class",
    ratios=(0.70, 0.15, 0.15),
    seed: int = 42,
    record_id_col: str = "record_id",
    text_col: str = "anamnesa",
    source_col: str = "source",
):
    """Split V4 tanpa membuang baris, grouped by pasien dan template teks.

    Greedy assignment menjaga distribusi gabungan ``class x source`` sedekat
    mungkin dengan target 70/15/15. Ini menggantikan pola lama "split baris
    lalu buang leak", yang menghilangkan 584/5.249 baris V3.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios harus total 1.0"
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    work = df.reset_index(drop=True).copy()
    work["_group_id"] = _connected_group_ids(work, record_id_col, text_col)
    if source_col in work.columns:
        work["_stratum"] = work[class_col].astype(str) + "\x1f" + work[source_col].astype(str)
    else:
        work["_stratum"] = work[class_col].astype(str)

    strata = sorted(work["_stratum"].unique())
    stratum_index = {name: i for i, name in enumerate(strata)}
    totals = np.zeros(len(strata), dtype=float)
    for name, count in work["_stratum"].value_counts().items():
        totals[stratum_index[name]] = count

    group_vectors: dict[int, np.ndarray] = {}
    group_sizes: dict[int, int] = {}
    for group_id, group in work.groupby("_group_id", sort=False):
        vector = np.zeros(len(strata), dtype=float)
        for name, count in group["_stratum"].value_counts().items():
            vector[stratum_index[name]] = count
        group_vectors[int(group_id)] = vector
        group_sizes[int(group_id)] = len(group)

    rng = np.random.RandomState(seed)
    tie_break = {group_id: rng.random() for group_id in group_vectors}
    rarity = {
        group_id: max((1.0 / totals[i]) for i in np.flatnonzero(vector))
        for group_id, vector in group_vectors.items()
    }
    ordered_groups = sorted(
        group_vectors,
        key=lambda group_id: (-rarity[group_id], -group_sizes[group_id], tie_break[group_id]),
    )

    split_names = ("train", "val", "test")
    targets = np.outer(np.asarray(ratios, dtype=float), totals)
    target_rows = np.asarray(ratios, dtype=float) * len(work)
    assigned = np.zeros_like(targets)
    assigned_rows = np.zeros(3, dtype=float)
    assignments: dict[int, int] = {}

    def global_cost(counts: np.ndarray, row_counts: np.ndarray) -> float:
        class_source_cost = np.square((counts - targets) / np.sqrt(targets + 1.0)).sum()
        row_cost = np.square((row_counts - target_rows) / np.sqrt(target_rows + 1.0)).sum()
        return float(class_source_cost + 0.25 * row_cost)

    for group_id in ordered_groups:
        vector = group_vectors[group_id]
        size = group_sizes[group_id]
        candidates = []
        for split_idx in range(3):
            trial_counts = assigned.copy()
            trial_rows = assigned_rows.copy()
            trial_counts[split_idx] += vector
            trial_rows[split_idx] += size
            candidates.append((global_cost(trial_counts, trial_rows), split_idx))
        _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        assignments[group_id] = chosen
        assigned[chosen] += vector
        assigned_rows[chosen] += size

    parts = []
    for split_idx, split_name in enumerate(split_names):
        group_ids = {group_id for group_id, assigned_idx in assignments.items() if assigned_idx == split_idx}
        part = work[work["_group_id"].isin(group_ids)].drop(columns=["_group_id", "_stratum"])
        part = part.sample(frac=1, random_state=seed + split_idx).reset_index(drop=True)
        logger.info("stratified_group_split %s=%d baris", split_name, len(part))
        parts.append(part)

    assert sum(map(len, parts)) == len(df), "group split tidak boleh membuang baris"
    return tuple(parts)


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
