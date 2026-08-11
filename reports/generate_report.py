"""
Baca experiments/runs.jsonl, generate tabel laporan -- supaya angka di
laporan TA SELALU sinkron sama hasil eksperimen asli, tidak ada lagi
salin-tempel manual antar dokumen (yang kemarin jadi sumber inkonsistensi).
"""
import json


def load_runs(path: str = "experiments/runs.jsonl") -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_matrix_table(runs: list) -> str:
    """TODO: group runs by (data_version, model_name), ambil run terbaru
    per kombinasi, generate tabel Markdown persis format matriks di
    'Rancangan Pipeline -- Redo Penuh untuk TA'.
    """
    raise NotImplementedError


def main():
    runs = load_runs()
    print(build_matrix_table(runs))


if __name__ == "__main__":
    main()
