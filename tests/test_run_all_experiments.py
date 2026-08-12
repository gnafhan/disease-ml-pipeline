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
def isolated_working_directory(tmp_path, monkeypatch):
    """Jangan pernah menulis/menghapus experiments/runs.jsonl milik repo."""
    monkeypatch.chdir(tmp_path)
    yield


def _fake_train_run_factory(call_log, fail_on=()):
    def _fake(data_version, model_key, cfg_path="config/experiment.yaml", smoke_test=False):
        run_id = f"{data_version}_{model_key}"
        call_log.append(run_id)
        if run_id in fail_on:
            raise RuntimeError(f"simulated failure: {run_id}")
        from src.pipeline import log_run
        # smoke_test ikut disimpan di record, sama seperti src.train.run asli --
        # dipakai buat filter --push-hf (run smoke-test nggak boleh ke-push).
        record = {"run_id": run_id, "test_f1_macro": 0.5, "smoke_test": smoke_test}
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


def test_only_filter_accepts_v4_without_changing_legacy_default_matrix():
    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)):
        sys.argv = ["run_all_experiments.py", "--only", "v4_base,v4_large", "--no-generate-report"]
        rae.main()
    assert call_log == ["v4_base", "v4_large"]


def test_full_run_is_not_skipped_by_old_smoke_test_record():
    os.makedirs("experiments", exist_ok=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "v1_base", "smoke_test": True}) + "\n")

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base", "--skip-existing", "--no-generate-report"]
        rae.main()
    assert call_log == ["v1_base"]


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


def test_push_hf_pushes_each_new_run_to_its_own_dynamic_repo(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    pushed = []

    def _fake_push_run_to_hf(record, repo_base, private=False, cleanup=False, experiments_dir="experiments"):
        pushed.append((record["run_id"], repo_base))
        return f"{repo_base}-{record['run_id'].replace('_', '-')}"

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)), \
         mock.patch("src.push_to_hf.push_run_to_hf", side_effect=_fake_push_run_to_hf):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base,v2_large",
                    "--push-hf", "someuser/pkt-indobert", "--no-generate-report"]
        rae.main()

    assert pushed == [("v1_base", "someuser/pkt-indobert"), ("v2_large", "someuser/pkt-indobert")]


def test_push_hf_one_failure_does_not_stop_others(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")

    def _fake_push_run_to_hf(record, repo_base, private=False, cleanup=False, experiments_dir="experiments"):
        if record["run_id"] == "v1_base":
            raise RuntimeError("simulated HF network error")
        return f"{repo_base}-{record['run_id'].replace('_', '-')}"

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)), \
         mock.patch("src.push_to_hf.push_run_to_hf", side_effect=_fake_push_run_to_hf) as push_mock, \
         mock.patch("src.run_all_experiments.time.sleep"):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base,v2_large",
                    "--push-hf", "someuser/pkt-indobert", "--no-generate-report"]
        result = rae.main()

    assert len(result["ok"]) == 2, "kombinasi training tetap 2/2 berhasil walau push-hf v1_base gagal"
    assert result["failed"] == [{
        "run_id": "v1_base", "stage": "push_hf",
        "error": "simulated HF network error",
    }]
    assert push_mock.call_count == 4  # v1 tiga retry + v2 satu kali
    out = capsys.readouterr().out
    assert "GAGAL permanen 'v1_base'" in out
    assert "someuser/pkt-indobert-v2-large" in out


def test_push_hf_skips_smoke_test_runs(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    pushed = []

    def _fake_push_run_to_hf(record, repo_base, private=False, cleanup=False, experiments_dir="experiments"):
        pushed.append(record["run_id"])
        return f"{repo_base}-{record['run_id']}"

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)), \
         mock.patch("src.push_to_hf.push_run_to_hf", side_effect=_fake_push_run_to_hf):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base", "--smoke-test",
                    "--push-hf", "someuser/pkt-indobert", "--no-generate-report"]
        rae.main()

    assert pushed == [], "run smoke-test tidak boleh ikut di-push ke HuggingFace"


def test_push_hf_cleanup_flag_forwarded_to_push_run_to_hf(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    seen_cleanup = []

    def _fake_push_run_to_hf(record, repo_base, private=False, cleanup=False, experiments_dir="experiments"):
        seen_cleanup.append(cleanup)
        return f"{repo_base}-{record['run_id']}"

    call_log = []
    with mock.patch("src.train.run", _fake_train_run_factory(call_log)), \
         mock.patch("src.push_to_hf.push_run_to_hf", side_effect=_fake_push_run_to_hf):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base", "--push-hf", "someuser/pkt-indobert",
                    "--push-hf-cleanup-local", "--no-generate-report"]
        rae.main()

    assert seen_cleanup == [True], "--push-hf-cleanup-local harus diteruskan sbg cleanup=True"


def test_push_hf_happens_right_after_its_own_combo_not_batched_at_the_end(monkeypatch):
    # Ini nge-cek --push-hf jalan PER KOMBINASI (interleaved), bukan nunggu
    # semua kombinasi kelar dulu -- penting buat skenario disk /kaggle/working
    # penuh: kalau push+cleanup ditunda sampai akhir, kombinasi terakhir bisa
    # kehabisan tempat SEBELUM sempat push+cleanup jalan sama sekali.
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    order = []

    def _fake_train_run(data_version, model_key, cfg_path="config/experiment.yaml", smoke_test=False):
        run_id = f"{data_version}_{model_key}"
        order.append(("train", run_id))
        from src.pipeline import log_run
        record = {"run_id": run_id, "test_f1_macro": 0.5, "smoke_test": smoke_test}
        log_run(record)
        return record

    def _fake_push_run_to_hf(record, repo_base, private=False, cleanup=False, experiments_dir="experiments"):
        order.append(("push", record["run_id"]))
        return f"{repo_base}-{record['run_id']}"

    with mock.patch("src.train.run", _fake_train_run), \
         mock.patch("src.push_to_hf.push_run_to_hf", side_effect=_fake_push_run_to_hf):
        sys.argv = ["run_all_experiments.py", "--only", "v1_base,v2_large",
                    "--push-hf", "someuser/pkt-indobert", "--no-generate-report"]
        rae.main()

    assert order == [
        ("train", "v1_base"), ("push", "v1_base"),
        ("train", "v2_large"), ("push", "v2_large"),
    ], "push v1_base harus terjadi SEBELUM training v2_large dimulai, bukan setelah semuanya kelar"
