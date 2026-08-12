"""
Test logika src/push_to_hf.py -- pilih run terbaik, dedup run_id, dan validasi
argumen. Semua panggilan ke HuggingFace Hub asli (create_repo/upload_folder)
di-mock, jadi tidak butuh internet/token sungguhan buat jalanin test ini.
"""

import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.push_to_hf import build_model_card, load_runs, pick_best_run, push_best_model, select_run

RUNS_PATH = "experiments/runs.jsonl"


def _write_runs(records, path=RUNS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def setup_function(_):
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)


def teardown_function(_):
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)


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
    # v2_base menang walau test_f1_macro-nya lebih rendah, karena skor yang
    # dipakai buat milih adalah reliable_only (lebih jujur soal kelas minor)
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


def test_push_best_model_uploads_from_correct_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)

    _write_runs([
        {"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.5, "model_name": "m1"},
        {"run_id": "v2_large", "smoke_test": False, "test_f1_macro": 0.8, "model_name": "m2"},
    ])
    model_dir = tmp_path / "experiments" / "v2_large"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake_api = mock.Mock()
    with mock.patch("huggingface_hub.HfApi", return_value=fake_api), \
         mock.patch("huggingface_hub.create_repo") as fake_create_repo:
        run = push_best_model("someuser/somerepo")

    assert run["run_id"] == "v2_large"  # skor tertinggi yang non-smoke-test
    fake_create_repo.assert_called_once()
    fake_api.upload_folder.assert_called_once()
    call_kwargs = fake_api.upload_folder.call_args.kwargs
    assert call_kwargs["repo_id"] == "someuser/somerepo"
    assert call_kwargs["folder_path"] == os.path.join("experiments", "v2_large")
    assert (model_dir / "README.md").exists()  # model card ke-generate


def test_push_best_model_raises_clear_error_without_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_runs([{"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.5}])

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        push_best_model("someuser/somerepo")


def test_push_best_model_raises_clear_error_when_model_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    monkeypatch.chdir(tmp_path)
    _write_runs([{"run_id": "v1_base", "smoke_test": False, "test_f1_macro": 0.5}])
    # folder experiments/v1_base/ SENGAJA tidak dibuat

    with pytest.raises(RuntimeError, match="tidak ketemu"):
        push_best_model("someuser/somerepo")
