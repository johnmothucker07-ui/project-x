"""Тесты подэксперимента о разрешении (шаги 4.3, 4.5)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.experiment import resolution as R  # noqa: E402


def checkerboard(px: int, square: int) -> np.ndarray:
    """Мелкая шахматка — фактура, которая обязана исчезнуть при огрублении."""
    rows = (np.arange(px) // square) % 2
    pattern = np.logical_xor.outer(rows, rows).astype(np.uint8) * 255
    return np.repeat(pattern[:, :, None], 3, axis=2)


# --- огрубление ---

def test_degrade_returns_encoder_size_whatever_the_level():
    """Все уровни приходят одного размера: иначе менялась бы и детализация,
    и размер входа, и разницу нельзя было бы приписать разрешению."""
    cell = checkerboard(1000, 10)
    for level in (0.5, 1.0, 3.0, 10.0):
        img = Image.open(io.BytesIO(R.degrade(cell, 500, level, 896)))
        assert img.size == (896, 896)


def test_degrade_actually_destroys_fine_texture():
    """Шахматка 5 м должна пережить 1 м и погибнуть на 10 м."""
    cell = checkerboard(1000, 10)          # клетка 500 м, квадрат 5 м

    def contrast(payload: bytes) -> float:
        return float(np.asarray(Image.open(io.BytesIO(payload))).std())

    fine = contrast(R.degrade(cell, 500, 1.0, 896))
    coarse = contrast(R.degrade(cell, 500, 10.0, 896))
    assert fine > 100          # шахматка ещё различима
    assert coarse < fine / 2   # на 10 м сливается в серое


def test_degrade_is_deterministic():
    """Иначе кэш ответов модели бесполезен и прогон невоспроизводим."""
    cell = checkerboard(400, 8)
    assert R.degrade(cell, 500, 3.0, 256) == R.degrade(cell, 500, 3.0, 256)


# --- меры ---

def frame(**by_level) -> pd.DataFrame:
    rows = []
    for level, scores in by_level.items():
        for n, value in enumerate(scores):
            rows.append({"cell_id": f"cell_{n}", "gsd_m": float(level.strip("g_")),
                         "score": value})
    return pd.DataFrame(rows)


def test_reference_agrees_with_itself_perfectly():
    data = frame(g_1=[1, 2, 3, 4, 5], g_3=[1, 2, 3, 4, 5])
    result = R.agreement(data, 1.0).set_index("gsd_m")
    assert result.loc[1.0, "spearman"] == 1.0
    assert result.loc[1.0, "mean_abs_shift"] == 0.0


def test_shift_and_correlation_are_reported():
    data = frame(g_1=[0, 1, 2, 3, 9], g_10=[0, 1, 2, 3, 0])
    row = R.agreement(data, 1.0).set_index("gsd_m").loc[10.0]
    assert row["mean_abs_shift"] == pytest.approx(9 / 5)
    assert row["share_shift_over_2"] == pytest.approx(1 / 5)
    assert row["max_shift"] == 9
    assert row["spearman"] < 1.0


def test_agreement_covers_every_level():
    data = frame(g_1=[1, 2, 3], g_3=[1, 2, 3], g_10=[3, 2, 1])
    assert sorted(R.agreement(data, 1.0)["gsd_m"]) == [1.0, 3.0, 10.0]


def test_noise_floor_uses_the_closest_pair():
    """Пол шума берётся там, где физической разницы почти нет, — иначе не с
    чем сравнивать эффект огрубления."""
    data = frame(g_1=[1, 2, 3, 4], g_3=[1, 2, 3, 4], g_10=[5, 5, 5, 5])
    floor = R.noise_floor(data)
    assert floor["pair"] == "1.0 vs 3.0"
    assert floor["mean_abs_shift"] == 0.0


def test_build_report_counts_skips():
    report = R.BuildReport(cells_total=10, cells_built=8,
                           skipped={"нет покрытия": 2})
    assert report.as_dict()["skipped"] == {"нет покрытия": 2}
