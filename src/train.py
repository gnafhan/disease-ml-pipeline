"""
Train -- Tahap 06. SATU entrypoint untuk semua model (base/large), dipilih
lewat argumen CLI -- gantiin v5_fixed.py / v6_training.py / v7_ensemble.py
yang tadinya file terpisah-pisah per eksperimen.

Usage:
    python -m src.train --data-version v3 --model large
    python -m src.train --data-version v3 --model base
"""
import argparse
from src.split import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", required=True, choices=["v1", "v2", "v3"])
    parser.add_argument("--model", required=True, choices=["base", "large"])
    args = parser.parse_args()

    config = load_config()
    hf_name = config["models"][args.model]["hf_name"]

    # TODO port dari v6_training.py:
    #   1. load data/processed/{data_version}/{train,val,test}.csv
    #   2. load tokenizer & model dari hf_name (config["training"])
    #   3. Focal Loss (gamma dari config) + label smoothing + class weights
    #   4. symptom keyword injection (opsional, kalau mau dipertahankan)
    #   5. training loop dengan early stopping (patience dari config)
    #   6. simpan checkpoint ke model_checkpoints/{data_version}_{model}/
    #      (folder ini di .gitignore -- upload manual ke HF Hub / Kaggle Dataset)
    raise NotImplementedError


if __name__ == "__main__":
    main()
