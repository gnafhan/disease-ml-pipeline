#!/usr/bin/env python3
"""
Entrypoint TIPIS buat kernel Kaggle jenis "script" (bukan notebook). Semua
logika training/evaluasi/push sudah ada dan ke-test di repo (src/*.py) --
file ini CUMA orkestrasi: tarik Secret, clone/pull repo terbaru, siapin
data, panggil src.run_all_experiments. JANGAN taruh logic ML di sini --
kalau mau ubah behavior training/evaluasi, edit src/train.py atau
src/run_all_experiments.py di GitHub, bukan di sini.

Kenapa dibikin gini (bukan notebook manual kayak sebelumnya): kernel jenis
"script" dieksekusi Kaggle sebagai batch job -- push, jalan sampai selesai,
ambil output -- BUKAN sesi interaktif yang perlu di-klik cell satu-satu.
Ini yang bikin --push-git & --skip-existing (lihat run_all_experiments.py)
penting: setiap kali script ini di-push ulang, dia mulai dari container
BARU (working dir kosong), tapi otomatis lanjut dari kombinasi yang belum
selesai berdasarkan experiments/runs.jsonl paling baru di GitHub.

Cara pakai dari Terminal (BUKAN dari chat -- kredensial Kaggle jangan
pernah ditulis di chat):
    cd kaggle/
    kaggle kernels push -p .
    # tunggu sampai selesai (cek status di kaggle.com/code atau via
    # `kaggle kernels status <username>/<kernel-slug>`), baru ambil output:
    kaggle kernels output <username>/<kernel-slug> -p ./outputs

Prasyarat SEKALI SAJA sebelum push pertama kali:
  1. `kaggle auth login` (atau set token, lihat README) di Terminal Mac.
  2. Push sekali dulu (`kaggle kernels push -p kaggle/`) supaya kernel-nya
     TERBENTUK di akun Kaggle kamu.
  3. Buka kernel itu di kaggle.com/code, Add-ons -> Secrets, centang
     GITHUB_TOKEN, GITHUB_REPO, HF_TOKEN, HF_REPO_ID (Secrets nggak bisa
     di-set lewat CLI, ini satu-satunya langkah manual di web yang wajib).
  4. Push lagi -- kali ini beneran jalan training-nya.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

WORK_DIR = "/kaggle/working"
REPO_DIR = os.path.join(WORK_DIR, "repo")
DATASET_GLOB = "/kaggle/input/**/pkt-processed-data"
# Repo ini publik. Fallback ini hanya dipakai bila layanan Kaggle Secrets
# sementara tidak dapat dihubungi; Secret GITHUB_REPO tetap menang bila tersedia.
PUBLIC_GITHUB_REPO = "gnafhan/disease-ml-pipeline"
RUN_IDS = ["v3_base", "v3_large", "v4_base", "v4_large"]


def _try_secret(client: UserSecretsClient, name: str) -> str | None:
    try:
        return client.get_secret(name)
    except Exception as exc:
        # Jangan cetak nilai Secret. Nama dan tipe error cukup untuk membedakan
        # Secret belum di-attach dari masalah akses sementara di Kaggle.
        print(f"[kernel] Secret {name} tidak tersedia ({type(exc).__name__}).")
        return None


def load_secrets_into_env() -> None:
    secrets = UserSecretsClient()
    for name in ["GITHUB_TOKEN", "GITHUB_REPO", "HF_TOKEN", "HF_REPO_ID"]:
        value = _try_secret(secrets, name)
        if value:
            os.environ[name] = value
    os.environ.setdefault("GITHUB_REPO", PUBLIC_GITHUB_REPO)


def build_repo_url() -> str:
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO")
    if github_token:
        return f"https://{github_token}@github.com/{github_repo}.git"
    return f"https://github.com/{github_repo}.git"  # repo public, token nggak wajib buat clone


def clone_repo(repo_url: str) -> None:
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)  # setiap run kernel script mulai dari container baru,
        # tapi defensif aja kalau-kalau ada sisa dari testing lokal.
    print("[kernel] clone repo ...")
    subprocess.run(["git", "clone", repo_url, REPO_DIR], check=True)


def install_requirements() -> None:
    print("[kernel] pip install -r requirements.txt ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=REPO_DIR, check=True,
    )


def validate_hf_push_target() -> None:
    """Fail sebelum GPU bila Secret/token/repo Hugging Face tidak writable."""
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    repo_base = os.environ.get("HF_REPO_ID")
    if not token or not repo_base:
        raise RuntimeError(
            "HF_TOKEN dan HF_REPO_ID wajib tersedia. Training dibatalkan sebelum GPU "
            "agar model tidak selesai tanpa bisa di-push."
        )
    account = HfApi(token=token).whoami()
    for run_id in RUN_IDS:
        repo_id = f"{repo_base}-{run_id.replace('_', '-')}"
        create_repo(repo_id, token=token, private=True, exist_ok=True)
    print(f"[kernel] Hugging Face write access OK untuk {account.get('name', 'account')} "
          f"dan {len(RUN_IDS)} repo private.")


def stage_processed_data() -> None:
    matches = [m for m in glob.glob(DATASET_GLOB, recursive=True) if os.path.isdir(m)]
    assert matches, (
        f"'pkt-processed-data' tidak ketemu di /kaggle/input. Pastikan dataset itu "
        f"sudah ditambahkan ke kernel ini (Add Input). Isi /kaggle/input saat ini: "
        f"{os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else '(nggak ada)'}"
    )
    src = matches[0]
    print("[kernel] pakai sumber data:", src)
    dest = os.path.join(REPO_DIR, "data", "processed")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def run_experiments() -> None:
    """Smoke seluruh target dahulu, lalu train V3/V4 base+large saja."""
    only = ",".join(RUN_IDS)
    preflight = [
        sys.executable, "-m", "src.run_all_experiments",
        "--only", only, "--smoke-test", "--push-git",
    ]
    print("[kernel] preflight:", " ".join(preflight))
    subprocess.run(preflight, cwd=REPO_DIR, check=True)

    full = [
        sys.executable, "-m", "src.run_all_experiments",
        "--only", only, "--push-git",
        "--push-hf", os.environ["HF_REPO_ID"],
        "--push-hf-private", "--push-hf-cleanup-local",
    ]
    print("[kernel] full training:", " ".join(full))
    subprocess.run(full, cwd=REPO_DIR, check=True)


def export_results() -> None:
    """Salin artefak kecil yang dibutuhkan ke output kernel Kaggle.

    Folder experiments/ dan reports/ di repo di-ignore Git sehingga Kaggle
    tidak selalu memasukkannya saat output kernel diunduh. Salinan eksplisit
    ini menjaga runs.jsonl dan laporan final tetap bisa diambil setelah sesi
    selesai, bahkan saat Kaggle Secrets/GitHub backup sedang bermasalah.
    """
    destination = os.path.join(WORK_DIR, "final_results")
    os.makedirs(destination, exist_ok=True)
    for relative_path in ["experiments/runs.jsonl", "reports/matriks_perbandingan.md"]:
        source = os.path.join(REPO_DIR, relative_path)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(destination, os.path.basename(source)))
    # Kalau HuggingFace Secret gagal, model jangan ikut hilang saat kernel mati.
    # Move (bukan copy) supaya tidak menggandakan pemakaian disk Kaggle.
    for run_id in RUN_IDS:
        model_dir = os.path.join(REPO_DIR, "experiments", run_id)
        if os.path.isdir(model_dir):
            shutil.move(model_dir, os.path.join(destination, run_id))
    print(f"[kernel] hasil ringkas diekspor ke {destination}")


def main() -> None:
    load_secrets_into_env()
    repo_url = build_repo_url()
    clone_repo(repo_url)
    install_requirements()
    validate_hf_push_target()
    stage_processed_data()
    try:
        run_experiments()
    finally:
        export_results()
    print("[kernel] SELESAI. runs.jsonl & reports sudah ke-push ke GitHub "
          "(kalau --push-git aktif); model sudah ke-push ke HuggingFace "
          "(kalau --push-hf aktif). Cek juga output kernel ini via "
          "`kaggle kernels output` buat log lengkapnya.")


if __name__ == "__main__":
    main()
