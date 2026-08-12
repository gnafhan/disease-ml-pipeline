"""
Test orkestrasi src/run_all_experiments.py -- pakai src.train.run yang di-mock
(bukan training asli) supaya bisa jalan tanpa GPU/torch/internet, fokus
nge-test LOGIKA wrapper-nya: urutan 6 kombinasi, lanjut jalan walau 1 gagal,
--only filter, --skip-existing.
"""

import importlib
import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.run_all_experiments as rae

RUNS_PATH = "experiments/runs.jsonl"


@pytest.fixture(autouse=True)
def clean_runs_file():
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)
    yield
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)


def _fake_train_run_factory(call_log, fail_on=()):
    def _fake(data_version, model_key, cfg_path="config/experiment.yaml", smoke_test=False):
        run_id = f"{data_version}_{model_key}"
        call_log.append(run_id)
        if run_id in fail_on:
            raise RuntimeError(f"simulated failure: {run_id}")
        from src.pipeline import log_run
        record = {"run_id": run_id, "test_f1_macro": 0.5}
        log_run(record)
        return record
    return _fake


def test_runs_all_six_combos_in_order():
    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)):
        sys.argv = ["run_all_experiments.py", "--no-generate-report"]
        result = rae.main()
    assert call_log == ["v1_base", "v1_large", "v2_base", "v2_large", "v3_base", "v3_large"]
    assert len(result["ok"]) == 6
    assert result["failed"] == []


def test_continues_after_one_combo_fails():
    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log, fail_on={"v2_large"})):
        sys.argv = ["run_all_experiments.py", "--no-generate-report"]
        result = rae.main()
    # semua 6 tetap DIPANGGIL walau v2_large gagal -- bukan berhenti di tengah
    assert call_log == ["v1_base", "v1_large", "v2_base", "v2_large", "v3_base", "v3_large"]
    assert len(result["ok"]) == 5
    assert result["failed"] == [{"run_id": "v2_large", "error": "simulated failure: v2_large"}]


def test_only_filter_runs_subset():
    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)):
        sys.argv = ["run_all_experiments.py", "--only", "v1_large,v3_base", "--no-generate-report"]
        rae.main()
    assert call_log == ["v1_large", "v3_base"]


def test_only_filter_rejects_unknown_run_id():
    with pytest.raises(SystemExit):
        sys.argv = ["run_all_experiments.py", "--only", "v9_huge", "--no-generate-report"]
        rae.main()


def test_skip_existing_only_runs_new_combos():
    os.makedirs("experiments", exist_ok=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "v1_base"}) + "\n")
        f.write(json.dumps({"run_id": "v1_large"}) + "\n")

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)):
        sys.argv = ["run_all_experiments.py", "--skip-existing", "--no-generate-report"]
        result = rae.main()

    assert call_log == ["v2_base", "v2_large", "v3_base", "v3_large"]
    assert len(result["ok"]) == 4


def test_push_git_skips_quietly_without_env_vars(monkeypatch, capsys):
    # Tanpa GITHUB_TOKEN/GITHUB_REPO di environment, --push-git harus SKIP
    # dengan pesan, bukan crash -- training tidak boleh berhenti gara-gara
    # backup gagal.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base", "--push-git", "--no-generate-report"]
        result = rae.main()
    assert len(result["ok"]) == 1
    assert "[push-git] SKIP" in capsys.readouterr().out


def test_push_git_calls_commit_and_push_after_each_combo(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_REPO", "gnafhan/disease-ml-pipeline")

    subprocess_calls = []

    def _fake_run(cmd, **kwargs):
        subprocess_calls.append(cmd)
        result = mock.Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)), \
         mock.patch("src.run_all_experiments.subprocess.run", side_effect=_fake_run):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base,v2_large", "--push-git", "--no-generate-report"]
        rae.main()

    push_calls = [c for c in subprocess_calls if c[:2] == ["git", "push"]]
    assert len(push_calls) == 2, "harus push SETELAH TIAP kombinasi (2 kombinasi -> 2x push), bukan cuma di akhir"
    assert all("dummy-token" in c[2] for c in push_calls)


def test_push_git_failure_does_not_stop_training(monkeypatch):
    # Push gagal (mis. network Kaggle lagi bermasalah) TIDAK BOLEH ngehentiin
    # training kombinasi berikutnya -- ini justru skenario paling penting
    # buat --push-git karena backup itu best-effort, bukan syarat lanjut.
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_REPO", "gnafhan/disease-ml-pipeline")

    def _fake_run(cmd, **kwargs):
        result = mock.Mock()
        if cmd[:2] == ["git", "push"]:
            result.returncode = 1
            result.stdout, result.stderr = "", "simulated network error"
        else:
            result.returncode = 0
            result.stdout, result.stderr = "", ""
        return result

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)), \
         mock.patch("src.run_all_experiments.subprocess.run", side_effect=_fake_run):
        sys.argv = ["run_all_experiments.py", "--push-git", "--no-generate-report"]
        result = rae.main()

    assert len(result["ok"]) == 6, "push gagal tidak boleh bikin kombinasi berikutnya batal dijalankan"
