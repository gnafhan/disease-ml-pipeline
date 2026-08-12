"""
Test logika src/push_to_hf.py -- pilih run, nama repo dinamis per run_id,
dedup run_id, dan push banyak model sekaligus (1 gagal tidak menghentikan
yang lain). Semua panggilan ke HuggingFace Hub asli (create_repo/upload_folder)
di-mock, jadi tidak butuh internet/token sungguhan buat jalanin test ini.
"""

import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.push_to_hf import (
    build_model_card,
    dynamic_repo_id,
    load_runs,
    pick_best_run,
    push_run_to_hf,
    push_to_hf,
    select_run,
)

RUNS_PATH = "experiments/runs.jsonl"


def _write_runs(records, path=RUNS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    """Hugging Face unit tests must not mutate the repo's real run log."""
    monkeypatch.chdir(tmp_path)
    yield


def test_dynamic_repo_id_replaces_underscore_with_hyphen():
    assert dynamic_repo_id("gnafhan/pkt-indobert", "v1_base") == "gnafhan/pkt-indobert-v1-base"
    assert dynamic_repo_id("gnafhan/pkt-indobert", "v3_large") == "gnafhan/pkt-indobert-v3-large"


def test_pick_best_run_ignores_smoke_test_entries():
    runs = [
        {"run_id": "v1_base", "smoke_test": True, "test_f1_macro": 0.99},  # smoke -- HARUS diabaikan
        {"run_id": "v2_base", "smoke_test": False, "test_f1_macro": 0.60},
        {"run_id": "v3_large", "smoke_test": False, "test_f1_macro": 0.70},
    ]
    best = pick_best_run(runs)
    assert best["run_id"] == "v3_large"


def test_pick_best_run_prefers_reliable_only_metric_when_present():
    runs = [
        {"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.80, "test_f1_macro_reliable_only": 0.50},
        {"run_id": "v2_base", "smoke_test": False, "test_f1_macro": 0.60, "test_f1_macro_reliable_only": 0.65},
    ]
    best = pick_best_run(runs)
    assert best["run_id"] == "v2_base"


def test_pick_best_run_raises_when_only_smoke_test_available():
    runs = [{"run_id": "v1_base", "smoke_test": True, "test_f1_macro": 0.9}]
    with pytest.raises(SystemExit):
        pick_best_run(runs)


def test_select_run_by_explicit_run_id():
    runs = [
        {"run_id": "v1_base", "test_f1_macro": 0.5},
        {"run_id": "v2_base", "test_f1_macro": 0.9},
    ]
    selected = select_run(runs, "v1_base")
    assert selected["run_id"] == "v1_base"  # bukan otomatis ambil yang skor tertinggi


def test_select_run_unknown_run_id_raises():
    runs = [{"run_id": "v1_base", "test_f1_macro": 0.5}]
    with pytest.raises(SystemExit):
        select_run(runs, "v9_huge")


def test_load_runs_dedups_keeping_last_entry_per_run_id():
    _write_runs([
        {"run_id": "v1_base", "test_f1_macro": 0.1},
        {"run_id": "v1_base", "test_f1_macro": 0.9},  # entri terbaru
    ])
    runs = load_runs()
    assert len(runs) == 1
    assert runs[0]["test_f1_macro"] == 0.9


def test_build_model_card_includes_key_metrics_and_disclaimer():
    run = {
        "run_id": "v3_large", "model_name": "indobenchmark/indobert-large-p1",
        "data_version": "v3", "num_classes": 12,
        "test_accuracy": 0.838, "test_f1_macro": 0.701, "test_f1_macro_reliable_only": 0.70,
    }
    card = build_model_card(run)
    assert "v3_large" in card
    assert "0.7010" in card or "0.701" in card
    assert "BUKAN alat diagnosis" in card and "klinis definitif" in card


def test_push_run_to_hf_uses_dynamic_repo_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    model_dir = tmp_path / "experiments" / "v2_large"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    run = {"run_id": "v2_large", "test_f1_macro": 0.8, "model_name": "m2"}
    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo") as fake_create_repo:
        repo_id = push_run_to_hf(run, "someuser/pkt-indobert")

    assert repo_id == "someuser/pkt-indobert-v2-large"
    fake_create_repo.assert_called_once_with(
        "someuser/pkt-indobert-v2-large", token="dummy-hf-token", private=False, exist_ok=True
    )
    fake_api.upload_folder.assert_called_once()
    assert fake_api.upload_folder.call_args.kwargs["repo_id"] == "someuser/pkt-indobert-v2-large"
    assert (model_dir / "README.md").exists()


def test_push_run_to_hf_cleanup_removes_local_model_dir_after_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    model_dir = tmp_path / "experiments" / "v1_base"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("fake weights")

    run = {"run_id": "v1_base", "test_f1_macro": 0.5}
    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo"):
        push_run_to_hf(run, "someuser/pkt-indobert", cleanup=True)

    assert not model_dir.exists(), "cleanup=True harus hapus folder model lokal setelah push sukses"


def test_push_run_to_hf_without_cleanup_keeps_local_model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    model_dir = tmp_path / "experiments" / "v1_base"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_text("fake weights")

    run = {"run_id": "v1_base", "test_f1_macro": 0.5}
    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo"):
        push_run_to_hf(run, "someuser/pkt-indobert")  # cleanup default False

    assert model_dir.exists(), "tanpa cleanup, folder model lokal harus TETAP ADA"


def test_push_run_to_hf_raises_clear_error_without_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    run = {"run_id": "v1_base", "test_f1_macro": 0.5}
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        push_run_to_hf(run, "someuser/pkt-indobert")


def test_push_run_to_hf_raises_clear_error_when_model_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)
    run = {"run_id": "v1_base", "test_f1_macro": 0.5}
    with pytest.raises(RuntimeError, match="tidak ketemu"):
        push_run_to_hf(run, "someuser/pkt-indobert")


def test_push_to_hf_default_pushes_every_non_smoke_test_run_to_own_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    _write_runs([
        {"run_id": "v1_base", "smoke_test": True, "test_f1_macro": 0.99},  # diabaikan (smoke)
        {"run_id": "v2_base", "smoke_test": False, "test_f1_macro": 0.60},
        {"run_id": "v3_large", "smoke_test": False, "test_f1_macro": 0.70},
    ])
    for run_id in ["v2_base", "v3_large"]:
        (tmp_path / "experiments" / run_id).mkdir(parents=True)
        (tmp_path / "experiments" / run_id / "config.json").write_text("{}")

    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo"):
        results = push_to_hf("someuser/pkt-indobert")

    assert {r["run_id"] for r in results} == {"v2_base", "v3_large"}  # smoke-test TIDAK ikut
    assert all(r["ok"] for r in results)
    assert {r["repo_id"] for r in results} == {
        "someuser/pkt-indobert-v2-base", "someuser/pkt-indobert-v3-large",
    }


def test_push_to_hf_one_failure_does_not_stop_the_others(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    _write_runs([
        {"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.5},
        {"run_id": "v2_base", "smoke_test": False, "test_f1_macro": 0.6},
    ])
    # cuma v2_base yang punya folder model -- v1_base sengaja dibiarkan gagal
    (tmp_path / "experiments" / "v2_base").mkdir(parents=True)
    (tmp_path / "experiments" / "v2_base" / "config.json").write_text("{}")

    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo"):
        results = push_to_hf("someuser/pkt-indobert")

    by_id = {r["run_id"]: r for r in results}
    assert by_id["v1_base"]["ok"] is False
    assert "tidak ketemu" in by_id["v1_base"]["error"]
    assert by_id["v2_base"]["ok"] is True  # tetap berhasil walau v1_base gagal


def test_push_to_hf_best_only_pushes_single_highest_scoring_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    _write_runs([
        {"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.5},
        {"run_id": "v3_large", "smoke_test": False, "test_f1_macro": 0.9},
    ])
    for run_id in ["v1_base", "v3_large"]:
        (tmp_path / "experiments" / run_id).mkdir(parents=True)

    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo"):
        results = push_to_hf("someuser/pkt-indobert", best_only=True)

    assert len(results) == 1
    assert results[0]["run_id"] == "v3_large"
