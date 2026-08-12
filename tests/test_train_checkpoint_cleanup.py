"""
Test src/train.py::cleanup_old_checkpoints -- reaksi dari kasus nyata di
Kaggle: kombinasi ke-6 (v3_large) gagal `OSError: No space left on device`
pas nyimpen model, karena checkpoint-XXXX/ bikinan Trainer (sampai 2 salinan
model penuh per kombinasi, save_total_limit=2) numpuk terus tanpa
dibersihkan sepanjang 6 kombinasi. Test ini murni operasi filesystem, TIDAK
butuh torch/GPU/model asli -- makanya bisa realistically dites di sini
walau logic training penuhnya sendiri cuma bisa diverifikasi di Kaggle.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import cleanup_old_checkpoints


def _make_dir_with_file(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.safetensors"), "w", encoding="utf-8") as f:
        f.write("fake weights")


def test_removes_all_checkpoint_dirs(tmp_path):
    output_dir = str(tmp_path / "v3_large")
    _make_dir_with_file(os.path.join(output_dir, "checkpoint-100"))
    _make_dir_with_file(os.path.join(output_dir, "checkpoint-200"))
    # file model FINAL di root -- ini yang HARUS TETAP ADA, bukan checkpoint
    _make_dir_with_file(output_dir)

    removed = cleanup_old_checkpoints(output_dir)

    assert len(removed) == 2
    assert not os.path.exists(os.path.join(output_dir, "checkpoint-100"))
    assert not os.path.exists(os.path.join(output_dir, "checkpoint-200"))
    # root output_dir (model final) TIDAK ikut terhapus, cuma checkpoint-*/
    assert os.path.exists(os.path.join(output_dir, "model.safetensors"))


def test_no_checkpoints_returns_empty_list_without_error(tmp_path):
    output_dir = str(tmp_path / "v1_base")
    _make_dir_with_file(output_dir)  # cuma model final, nggak ada checkpoint-*/

    removed = cleanup_old_checkpoints(output_dir)

    assert removed == []
    assert os.path.exists(os.path.join(output_dir, "model.safetensors"))


def test_missing_output_dir_returns_empty_list_without_raising(tmp_path):
    # output_dir yang belum pernah dibuat sama sekali -- glob() aman, cuma
    # balik list kosong, TIDAK raise FileNotFoundError.
    removed = cleanup_old_checkpoints(str(tmp_path / "tidak-ada"))
    assert removed == []


def test_only_deletes_dirs_matching_checkpoint_prefix(tmp_path):
    # folder lain yang KEBETULAN ada di output_dir (bukan checkpoint-*/)
    # HARUS tidak ikut terhapus -- cuma yang match prefix "checkpoint-".
    output_dir = str(tmp_path / "v2_base")
    _make_dir_with_file(os.path.join(output_dir, "checkpoint-50"))
    _make_dir_with_file(os.path.join(output_dir, "runs"))  # folder TensorBoard, bukan checkpoint

    removed = cleanup_old_checkpoints(output_dir)

    assert removed == [os.path.join(output_dir, "checkpoint-50")]
    assert os.path.exists(os.path.join(output_dir, "runs"))
