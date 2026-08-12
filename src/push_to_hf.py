"""
Push model hasil training ke HuggingFace Hub. Karena ada 6 kombinasi
(data-v1/v2/v3 x model-base/large), setiap kombinasi otomatis dapat REPO
SENDIRI di HuggingFace Hub -- nama repo diturunkan otomatis dari satu
"repo dasar" + run_id, mis. repo dasar "gnafhan/pkt-indobert" ->
"gnafhan/pkt-indobert-v1-base", "gnafhan/pkt-indobert-v3-large", dst. Kamu
CUMA perlu tentuin nama dasarnya sekali, sisanya otomatis -- nggak perlu
bikin 6 repo manual satu-satu di web HuggingFace (create_repo(exist_ok=True)
yang bikinnya kalau belum ada).

Butuh:
  - env var HF_TOKEN (HuggingFace User Access Token, scope "Write") -- di
    Kaggle simpan sbg Secret lalu di Cell:
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  - huggingface_hub sudah terinstall (ada di requirements.txt)
  - folder experiments/<run_id>/ berisi model+tokenizer hasil src/train.py
    (trainer.save_model() + tokenizer.save_pretrained() dipanggil otomatis
    di akhir src/train.py::run(), TIDAK perlu langkah manual tambahan)

Cara pakai:
    # push SEMUA run non-smoke-test yang ada di runs.jsonl, masing2 ke repo sendiri
    python -m src.push_to_hf --repo-base gnafhan/pkt-indobert

    # cuma push 1 kombinasi tertentu
    python -m src.push_to_hf --repo-base gnafhan/pkt-indobert --run-id v3_large

    # cuma push yang skornya (test_f1_macro_reliable_only) paling tinggi
    python -m src.push_to_hf --repo-base gnafhan/pkt-indobert --best-only

    python -m src.push_to_hf --repo-base gnafhan/pkt-indobert --private

Juga dipanggil otomatis dari src/run_all_experiments.py lewat
--push-hf <repo-base> setelah SEMUA kombinasi di run itu selesai -- jadi
satu command Kaggle: data sudah diproses -> training 6 kombinasi -> 6
model live di HuggingFace Hub (masing2 repo sendiri), tanpa langkah manual.
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


def dynamic_repo_id(repo_base: str, run_id: str) -> str:
    """
    'gnafhan/pkt-indobert' + 'v1_base' -> 'gnafhan/pkt-indobert-v1-base'.
    Underscore di run_id diganti "-" (konvensi nama repo HuggingFace lebih
    umum pakai "-", walau underscore juga valid).
    """
    return f"{repo_base}-{run_id.replace('_', '-')}"


def build_model_card(run: dict, repo_id: str | None = None) -> str:
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

Salah satu dari 6 kombinasi (data-v1/v2/v3 x model-base/large) di pipeline
[disease-ml-pipeline](https://github.com/gnafhan/disease-ml-pipeline) --
tiap kombinasi punya repo HuggingFace sendiri (lihat repo lain dgn prefix
nama yang sama utk perbandingan versi data/model lainnya).

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


def push_run_to_hf(run: dict, repo_base: str, private: bool = False,
                    experiments_dir: str = "experiments") -> str:
    """
    Push SATU run ke repo HuggingFace yang namanya diturunkan otomatis
    (lihat dynamic_repo_id). Return repo_id yang dipakai.
    """
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Env var HF_TOKEN belum di-set (HuggingFace User Access Token, scope Write).")

    model_dir = os.path.join(experiments_dir, run["run_id"])
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"Folder model '{model_dir}' tidak ketemu. Model cuma ada di working directory "
            "sesi Kaggle yang melatihnya -- kalau sesi itu sudah mati, run_id ini tidak bisa "
            "di-push lagi (retrain dulu)."
        )

    repo_id = dynamic_repo_id(repo_base, run["run_id"])

    card_path = os.path.join(model_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(build_model_card(run, repo_id))

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
    return repo_id


def push_to_hf(repo_base: str, run_id: str | None = None, best_only: bool = False,
               private: bool = False, runs_path: str = RUNS_PATH,
               experiments_dir: str = "experiments") -> list[dict]:
    """
    Push satu/beberapa/semua run ke HuggingFace Hub, masing2 ke repo
    dinamisnya sendiri. Return list of {"run_id", "repo_id", "ok", ["error"]}
    -- satu run gagal TIDAK menghentikan push run lainnya.
    """
    runs = load_runs(runs_path)
    if run_id:
        targets = [select_run(runs, run_id)]
    elif best_only:
        targets = [pick_best_run(runs)]
    else:
        targets = [r for r in runs if not r.get("smoke_test") and r.get("test_f1_macro") is not None]
        if not targets:
            raise SystemExit(
                f"Tidak ada run non-smoke-test di {runs_path}. Jalankan training asli dulu."
            )

    results = []
    for run in targets:
        try:
            repo_id = push_run_to_hf(run, repo_base, private=private, experiments_dir=experiments_dir)
            results.append({"run_id": run["run_id"], "repo_id": repo_id, "ok": True})
        except Exception as e:
            results.append({"run_id": run["run_id"], "error": str(e), "ok": False})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-base", required=True,
                         help="mis. gnafhan/pkt-indobert -- tiap run di-push ke "
                              "'<repo-base>-<run_id>' (run_id pakai '-', bukan '_')")
    parser.add_argument("--run-id", default=None,
                         help="Push run_id tertentu aja (mis. 'v3_large'). Default: SEMUA run "
                              "non-smoke-test, masing2 ke repo sendiri.")
    parser.add_argument("--best-only", action="store_true",
                         help="Cuma push run dgn test_f1_macro_reliable_only tertinggi.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--runs-path", default=RUNS_PATH)
    args = parser.parse_args()

    results = push_to_hf(args.repo_base, run_id=args.run_id, best_only=args.best_only,
                          private=args.private, runs_path=args.runs_path)
    for r in results:
        if r["ok"]:
            print(f"OK    {r['run_id']} -> https://huggingface.co/{r['repo_id']}")
        else:
            print(f"GAGAL {r['run_id']}: {r['error']}")


if __name__ == "__main__":
    main()
