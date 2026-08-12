"""
Push model hasil training ke HuggingFace Hub. Default: pilih otomatis run
dengan test_f1_macro_reliable_only (fallback test_f1_macro) TERTINGGI di
experiments/runs.jsonl -- run smoke-test (smoke_test=True) selalu diabaikan
karena metriknya nggak berarti apa-apa (lihat catatan di run_all_experiments.py).

Butuh:
  - env var HF_TOKEN (HuggingFace User Access Token, scope "Write") -- di
    Kaggle simpan sbg Secret lalu di Cell:
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  - huggingface_hub sudah terinstall (ada di requirements.txt)
  - folder experiments/<run_id>/ berisi model+tokenizer hasil src/train.py
    (trainer.save_model() + tokenizer.save_pretrained() dipanggil otomatis
    di akhir src/train.py::run(), TIDAK perlu langkah manual tambahan)

Cara pakai:
    python -m src.push_to_hf --repo-id gnafhan/pkt-indobert-best
    python -m src.push_to_hf --repo-id gnafhan/pkt-indobert-best --run-id v3_large
    python -m src.push_to_hf --repo-id gnafhan/pkt-indobert-best --private

Juga bisa dipanggil otomatis dari src/run_all_experiments.py lewat
--push-hf <repo-id> setelah SEMUA kombinasi selesai, jadi satu command
Kaggle bisa langsung: data sudah diproses -> training 6 kombinasi -> model
terbaik live di HuggingFace Hub, tanpa langkah manual terpisah.
"""

from __future__ import annotations

import argparse
import json
import os

RUNS_PATH = "experiments/runs.jsonl"


def load_runs(path: str = RUNS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"{path} tidak ketemu -- belum ada training yang selesai.")
    runs_by_id: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            run_id = record.get("run_id")
            if run_id:
                runs_by_id[run_id] = record  # entri terakhir per run_id menang, sama kayak generate_report.py
    return list(runs_by_id.values())


def pick_best_run(runs: list[dict]) -> dict:
    """
    Run smoke-test SELALU diabaikan (f1-nya nggak representatif, cuma 1
    epoch/2 sampel per kelas). Kalau semua run yang ada cuma smoke-test,
    gagal dengan pesan jelas -- jangan sampai kepilih diam-diam.
    """
    candidates = [
        r for r in runs
        if not r.get("smoke_test") and r.get("test_f1_macro") is not None
    ]
    if not candidates:
        raise SystemExit(
            "Tidak ada run TRAINING ASLI (non-smoke-test) dengan test_f1_macro di "
            f"{RUNS_PATH}. Jalankan `python -m src.run_all_experiments` (tanpa "
            "--smoke-test) dulu."
        )

    def score(r: dict) -> float:
        return r.get("test_f1_macro_reliable_only", r["test_f1_macro"])

    return max(candidates, key=score)


def select_run(runs: list[dict], run_id: str | None) -> dict:
    if run_id is None:
        return pick_best_run(runs)
    matches = [r for r in runs if r.get("run_id") == run_id]
    if not matches:
        raise SystemExit(f"run_id '{run_id}' tidak ketemu di {RUNS_PATH}. "
                          f"Yang ada: {sorted({r.get('run_id') for r in runs})}")
    return matches[-1]


def build_model_card(run: dict) -> str:
    def fmt(key):
        val = run.get(key)
        return f"{val:.4f}" if isinstance(val, (int, float)) else "?"

    return f"""---
language: id
license: unknown
tags:
- indobert
- text-classification
- indonesian
- medical
---

# {run.get('run_id', '?')} -- Klasifikasi Penyakit dari Anamnesa (PKT)

Model klasifikasi penyakit dari teks anamnesa Bahasa Indonesia (skrining awal
/ surveilans), fine-tuned dari `{run.get('model_name', '?')}`.

Dihasilkan otomatis oleh `src/push_to_hf.py` dari pipeline
[disease-ml-pipeline](https://github.com/gnafhan/disease-ml-pipeline).

## Data

- Versi data: `{run.get('data_version', '?')}`
- Jumlah kelas: {run.get('num_classes', '?')}
- n_train={run.get('n_train', '?')}, n_val={run.get('n_val', '?')}, n_test={run.get('n_test', '?')}

## Metrik (test set)

| Metrik | Nilai |
|---|---|
| Accuracy | {fmt('test_accuracy')} |
| Precision macro | {fmt('test_precision_macro')} |
| Recall macro | {fmt('test_recall_macro')} |
| F1 macro | {fmt('test_f1_macro')} |
| F1 macro (kelas reliable saja, support>=30) | {fmt('test_f1_macro_reliable_only')} |
| F1 weighted | {fmt('test_f1_weighted')} |

## PENTING -- batasan

Mapping label ICD-10 -> kelas direkonstruksi dari data historis rumah sakit
(lihat README repo pipeline), BELUM divalidasi ulang oleh dokter sumber data
secara formal. Model ini untuk skrining/riset awal, BUKAN alat diagnosis
klinis definitif. Jangan dipakai sebagai satu-satunya dasar keputusan medis.
"""


def push_best_model(repo_id: str, run_id: str | None = None, private: bool = False,
                     runs_path: str = RUNS_PATH, experiments_dir: str = "experiments") -> dict:
    """
    Push model dari experiments/<run_id>/ ke HuggingFace Hub `repo_id`.
    Return record run yang di-push (biar caller bisa log/print ringkasan).
    """
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Env var HF_TOKEN belum di-set (HuggingFace User Access Token, scope Write).")

    runs = load_runs(runs_path)
    run = select_run(runs, run_id)

    model_dir = os.path.join(experiments_dir, run["run_id"])
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Folder model '{model_dir}' tidak ketemu. Model cuma ada di working directory "
            "sesi Kaggle yang melatihnya -- kalau sesi itu sudah mati, run_id ini tidak bisa "
            "di-push lagi (retrain dulu)."
        )

    card_path = os.path.join(model_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(build_model_card(run))

    create_repo(repo_id, token=token, private=private, exist_ok=True)

    api = HfApi(token=token)
    headline = run.get("test_f1_macro_reliable_only", run.get("test_f1_macro"))
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "README.md", "*.model"],
        ignore_patterns=["checkpoint-*/**", "runs/**"],
        commit_message=f"Push {run['run_id']} (test_f1_macro_reliable_only={headline})",
    )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="mis. gnafhan/pkt-indobert-best")
    parser.add_argument("--run-id", default=None,
                         help="Push run_id tertentu (mis. 'v3_large'). Default: otomatis pilih "
                              "test_f1_macro_reliable_only tertinggi.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--runs-path", default=RUNS_PATH)
    args = parser.parse_args()

    run = push_best_model(args.repo_id, run_id=args.run_id, private=args.private,
                           runs_path=args.runs_path)
    headline = run.get("test_f1_macro_reliable_only", run.get("test_f1_macro"))
    print(f"Selesai push '{run['run_id']}' (test_f1_macro_reliable_only={headline}) -> "
          f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
