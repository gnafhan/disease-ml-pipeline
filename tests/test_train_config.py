"""
Regression test: pastikan setiap key yang dibaca src/train.py dari
config/experiment.yaml benar-benar ada di skema YAML-nya.

Ini reaksi dari bug nyata: train.py sempat baca cfg["split"]["seed"] padahal
"seed" ada di top-level config (cfg["seed"]), bukan nested di dalam "split".
Karena src/run_all_experiments.py di test_run_all_experiments.py selalu
nge-mock src.train.run secara penuh, bug ini nggak ketauan lewat test manapun
sampai dijalanin sungguhan di Kaggle -- makanya semua 6 kombinasi gagal
serentak dengan KeyError yang sama persis. Test ini nutup gap itu tanpa perlu
torch/GPU/HF download: cukup load config asli + cek strukturnya cocok sama
yang diasumsikan train.py, PAKAI SOURCE STRING train.py biar kalau ada
key baru yang dibaca tapi belum ada di config, ketauan dari test ini juga
(bukan cuma daftar manual yang bisa basi).
"""

import ast
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_real_config():
    with open(os.path.join(REPO_ROOT, "config", "experiment.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_has_top_level_seed_not_nested_under_split():
    cfg = _load_real_config()
    assert "seed" in cfg, "config/experiment.yaml harus punya key top-level 'seed'"
    assert isinstance(cfg["seed"], int)
    # split cuma boleh berisi rasio/strategy, BUKAN seed -- kalau ada yang nambahin
    # 'seed' lagi di dalam split tanpa update train.py, minimal ketauan di sini
    assert "seed" not in cfg.get("split", {}), (
        "kalau 'seed' ditambahkan di dalam 'split', pastikan src/train.py juga "
        "di-update supaya baca cfg['split']['seed'], jangan cfg['seed'] lagi"
    )


def test_config_has_models_and_training_keys_used_by_train_py():
    cfg = _load_real_config()
    for model_key in ("base", "large"):
        assert model_key in cfg["models"], f"config kurang models.{model_key}"
        assert "hf_name" in cfg["models"][model_key]

    for key in ("max_len", "batch_size", "grad_accum", "lr", "epochs",
                "early_stopping_patience", "focal_gamma"):
        assert key in cfg["training"], f"config['training'] kurang key '{key}'"

    assert "processed_dir" in cfg["paths"]
    assert isinstance(cfg["classes"], list) and len(cfg["classes"]) > 0
    assert "min_support_reliable" in cfg


def _extract_cfg_bracket_chains(source: str) -> list[list[str]]:
    """
    Cari semua pola cfg["a"]["b"]... di source train.py, balikin tiap chain
    sbg list of string key, mis. 'cfg["split"]["seed"]' -> ["split", "seed"].
    Regex sederhana (bukan full AST resolve) tapi cukup buat nangkep pola
    literal string yang dipakai di train.py saat ini.
    """
    pattern = re.compile(r'cfg((?:\["[^"]+"\])+)')
    chains = []
    for match in pattern.finditer(source):
        keys = re.findall(r'\["([^"]+)"\]', match.group(1))
        chains.append(keys)
    return chains


def test_every_cfg_key_chain_in_train_py_resolves_against_real_config():
    """
    Baca literal src/train.py, ekstrak semua akses cfg["x"]["y"] yang ada,
    lalu coba resolve satu-satu ke config/experiment.yaml asli. Kalau ada
    yang KeyError, test ini gagal DULUAN sebelum sempat ke Kaggle -- persis
    kasus 'seed' yang kemarin baru ketauan pas run_all_experiments jalan
    sungguhan di GPU.
    """
    cfg = _load_real_config()
    with open(os.path.join(REPO_ROOT, "src", "train.py"), encoding="utf-8") as f:
        source = f.read()

    chains = _extract_cfg_bracket_chains(source)
    assert chains, "tidak ada akses cfg[...] ditemukan di train.py -- cek regex/path"

    # model_key dipakai sbg key dinamis (cfg["models"][model_key]) -- ganti
    # dengan 'base'/'large' satu-satu biar bisa diresolve sbg literal.
    failures = []
    for keys in chains:
        for model_key in ("base", "large"):
            resolved_keys = [model_key if k == "model_key" else k for k in keys]
            node = cfg
            try:
                for k in resolved_keys:
                    node = node[k]
            except (KeyError, TypeError) as e:
                failures.append((keys, model_key, str(e)))

    assert not failures, (
        "Ada akses cfg[...] di train.py yang tidak match struktur "
        "config/experiment.yaml: " + "; ".join(
            f"cfg{''.join(f'[{k!r}]' for k in keys)} (model_key={mk}) -> {err}"
            for keys, mk, err in failures
        )
    )
