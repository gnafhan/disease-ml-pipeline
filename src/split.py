"""
Split -- Tahap 05. TERKUNCI. Seed & rasio SELALU dari config/experiment.yaml,
jangan pernah ditulis ulang/hardcode di file lain.
"""
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def load_config(path: str = "config/experiment.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stratified_split(df, label_col: str, config: dict):
    """70/15/15 stratified split, seed dari config['seed'].
    Return (train_df, val_df, test_df).

    TODO:
      1. filter df ke config['classes'] SEBELUM split (biar data-v1/v2 lama
         juga konsisten pakai 12 kelas final, bukan 17/14 kelas asli)
      2. train_test_split dua kali (train vs rest, lalu val vs test)
         dengan stratify=df[label_col] dan random_state=config['seed']
      3. assert tidak ada overlap index antar split (cegah data leakage
         yang dulu pernah kejadian)
    """
    raise NotImplementedError
