"""
Entrypoint tunggal untuk jalanin 6 kombinasi wajib (data-v1/v2/v3 x model-base/
large) di Kaggle -- ganti manual jalankan src/train.py 6x satu-satu.

Dirancang khusus utk kondisi Kaggle:
  - Kalau satu kombinasi error/crash (OOM, dsb), lanjut ke kombinasi berikutnya
    (bukan berhenti total) -- dicatat sbg failed, bukan diam-diam di-skip.
  - --skip-existing: kalau sesi Kaggle keputus (limit 9-12 jam) dan run ulang,
    kombinasi yang SUDAH ada di experiments/runs.jsonl otomatis dilewati,
    jadi tinggal lanjut dari yang belum kelar.
  - Print progress + estimasi sisa waktu per kombinasi (dari eksekusi
    sebelumnya) supaya kelihatan kalau bakal kelamaan sebelum GPU quota habis.
  - Generate reports/matriks_perbandingan.md otomatis di akhir kalau
    --generate-report (default: on) -- jadi begitu training kelar, tabel udah
    langsung ada, tidak perlu command terpisah.

Cara pakai (lihat README.md bagian "Cara run di Kaggle" utk tutorial lengkap):
    python -m src.run_all_experiments                     # jalankan 6 kombinasi
    python -m src.run_all_experiments --skip-existing      # lanjut dari yg blm ada
    python -m src.run_all_experiments --only v1_large,v3_base
    python -m src.run_all_experiments --smoke-test         # cek pipeline dulu (cepat)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

ALL_COMBOS = [
    ("v1", "base"), ("v1", "large"),
    ("v2", "base"), ("v2", "large"),
    ("v3", "base"), ("v3", "large"),
]

RUNS_PATH = "experiments/runs.jsonl"


def _load_existing_run_ids(runs_path: str = RUNS_PATH) -> set[str]:
    if not os.path.exists(runs_path):
        return set()
    ids = set()
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line).get("run_id"))
            except json.JSONDecodeError:
                continue
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                         help="Comma-separated run_id, mis. 'v1_large,v3_base'. Default: semua 6.")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Lewati kombinasi yang run_id-nya sudah ada di experiments/runs.jsonl")
    parser.add_argument("--smoke-test", action="store_true",
                         help="1 epoch, subset kecil per kelas -- utk cek semua kombinasi jalan tanpa error dulu")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--no-generate-report", action="store_true",
                         help="Jangan auto-generate reports/matriks_perbandingan.md di akhir")
    args = parser.parse_args()

    # Import di sini (bukan di top-level) supaya --help tetap jalan cepat
    # walau torch/transformers belum terinstall (mis. lagi cuma cek kombinasi).
    from src.train import run as train_run

    combos = ALL_COMBOS
    if args.only:
        wanted = set(args.only.split(","))
        combos = [(dv, mk) for dv, mk in ALL_COMBOS if f"{dv}_{mk}" in wanted]
        missing = wanted - {f"{dv}_{mk}" for dv, mk in combos}
        if missing:
            raise SystemExit(f"run_id tidak dikenal: {missing}. Pilihan valid: "
                              f"{[f'{dv}_{mk}' for dv, mk in ALL_COMBOS]}")

    existing = _load_existing_run_ids() if args.skip_existing else set()

    results, failures = [], []
    t_all = time.time()
    for i, (data_version, model_key) in enumerate(combos, 1):
        run_id = f"{data_version}_{model_key}"
        header = f"[{i}/{len(combos)}] {run_id}"

        if run_id in existing:
            print(f"{header} -- SKIP (sudah ada di runs.jsonl, pakai --skip-existing hilang kalau mau re-run)")
            continue

        print(f"{header} -- mulai training...")
        t0 = time.time()
        try:
            record = train_run(data_version, model_key, cfg_path=args.config, smoke_test=args.smoke_test)
            elapsed = time.time() - t0
            headline = record.get("test_f1_macro_reliable_only", record.get("test_f1_macro"))
            print(f"{header} -- SELESAI dalam {elapsed/60:.1f} menit. "
                  f"test_f1_macro_reliable_only={headline}")
            results.append(record)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{header} -- GAGAL setelah {elapsed/60:.1f} menit: {e}")
            traceback.print_exc()
            failures.append({"run_id": run_id, "error": str(e)})

        remaining = len(combos) - i
        if remaining and results:
            avg_min_per_combo = (time.time() - t_all) / 60 / i
            print(f"    (~{remaining} kombinasi lagi tersisa, rata-rata {avg_min_per_combo:.1f} menit/kombinasi sejauh ini "
                  f"-> estimasi ~{avg_min_per_combo * remaining:.0f} menit lagi)")

    total_min = (time.time() - t_all) / 60
    print(f"\n=== SELESAI: {len(results)}/{len(combos)} berhasil, {len(failures)} gagal, "
          f"total {total_min:.1f} menit ===")
    if failures:
        print("Yang gagal (cek traceback di atas, biasanya OOM -- coba turunkan batch_size di config):")
        for f in failures:
            print(f"  - {f['run_id']}: {f['error']}")

    if not args.no_generate_report:
        from reports.generate_report import main as generate_report_main
        generate_report_main()

    return {"ok": results, "failed": failures}


if __name__ == "__main__":
    main()
