"""
Pipeline -- Tahap 08 (orkestrasi end-to-end) + pencatatan runtime tiap tahap
(ini metrik engineering "runtime end-to-end" di rancangan).

Usage:
    python -m src.pipeline --data-version v3 --model large
    python -m src.pipeline --stage ingest   # cuma jalanin satu tahap, buat debug
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def log_run(record: dict, path: str = "experiments/runs.jsonl") -> None:
    """Append satu record ke runs.jsonl. record WAJIB minimal punya:
    run_id, timestamp, data_version, model_name, seed, class_list_hash,
    runtime_per_stage (dict), metrics (dari evaluate.compute_metrics).
    """
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", required=True, choices=["v1", "v2", "v3"])
    parser.add_argument("--model", required=True, choices=["base", "large"])
    args = parser.parse_args()

    runtime_per_stage = {}
    t0 = time.time()

    # TODO orkestrasi:
    #   ingest -> label -> clean -> split -> train -> evaluate
    #   catat time.time() di setiap batas tahap ke runtime_per_stage
    #   panggil log_run() di akhir dengan hasil lengkap

    raise NotImplementedError


if __name__ == "__main__":
    main()
