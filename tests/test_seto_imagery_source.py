"""Тесты чтения снимка под клетку (шаг 4.2).

Сеть не трогаем: плитки подделываем локальными GeoTIFF с той же геометрией,
что у swisstopo (1 км, EPSG:2056). Проверяем контракт — окно ровно по клетке,
склейка на стыке, отказ при неполном покрытии.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.imagery import source as S  # noqa: E402

GSD = 0.5          # в тестах берём крупный пиксель, чтобы файлы были лёгкими
TILE_M = 1000


def write_tile(path: Path, easting: int, northing: int, value: int) -> S.Tile:
    """Плитка 1 км, залитая одним цветом, с геопривязкой как у swisstopo."""
    side = int(TILE_M / GSD)
    data = np.full((3, side, side), value, dtype=np.uint8)
    transform = from_origin(easting, northing + TILE_M, GSD, GSD)
    with rasterio.open(path, "w", driver="GTiff", height=side, width=side,
                       count=3, dtype="uint8", crs="EPSG:2056",
                       transform=transform) as dst:
        dst.write(data)
    return S.Tile(href=str(path), year=2024,
                  bounds=(easting, northing, easting + TILE_M, northing + TILE_M))


@pytest.fixture
def one_tile(tmp_path):
    return [write_tile(tmp_path / "a.tif", 2662000, 1207000, 100)]


# --- геометрия ---

def test_covers_detects_overlap():
    tile = S.Tile("x", (2662000, 1207000, 2663000, 1208000), 2024)
    assert tile.covers((2662000, 1207000, 2662500, 1207500))
    assert not tile.covers((2663000, 1207000, 2663500, 1207500))


def test_touching_edge_is_not_overlap():
    """Клетка, начинающаяся ровно на границе плитки, лежит в соседней."""
    tile = S.Tile("x", (2662000, 1207000, 2663000, 1208000), 2024)
    assert not tile.covers((2661500, 1207000, 2662000, 1207500))


# --- чтение ---

def test_cell_inside_one_tile(one_tile):
    arr = S.read_cell(one_tile, (2662000, 1207000, 2662500, 1207500), 128)
    assert arr.shape == (128, 128, 3)
    assert (arr == 100).all()


def test_requested_size_is_respected(one_tile):
    """Размер выхода задаём мы, а не разрешение исходника."""
    for px in (64, 256, 512):
        assert S.read_cell(one_tile, (2662000, 1207000, 2662500, 1207500),
                           px).shape == (px, px, 3)


def test_cell_spanning_two_tiles_is_stitched(tmp_path):
    """Клетка на стыке должна собираться, а не обрезаться по краю плитки."""
    tiles = [write_tile(tmp_path / "l.tif", 2662000, 1207000, 50),
             write_tile(tmp_path / "r.tif", 2663000, 1207000, 200)]
    arr = S.read_cell(tiles, (2662750, 1207250, 2663250, 1207750), 100)
    left, right = arr[:, :50, 0], arr[:, 50:, 0]
    assert (left == 50).all() and (right == 200).all()


def test_partial_coverage_is_refused(tmp_path):
    """Чёрные поля исказили бы оценку сильнее, чем пропуск клетки."""
    tiles = [write_tile(tmp_path / "a.tif", 2662000, 1207000, 100)]
    with pytest.raises(S.CoverageError, match="покрыта снимком"):
        S.read_cell(tiles, (2662750, 1207250, 2663250, 1207750), 100)


def test_no_tiles_at_all_is_refused(one_tile):
    with pytest.raises(S.CoverageError, match="нет плиток"):
        S.read_cell(one_tile, (2700000, 1207000, 2700500, 1207500), 64)


def test_window_follows_the_cell_not_the_tile(tmp_path):
    """Разные половины плитки должны давать разное содержимое."""
    side = int(TILE_M / GSD)
    data = np.zeros((3, side, side), dtype=np.uint8)
    data[:, :, side // 2:] = 255                      # правая половина белая
    path = tmp_path / "half.tif"
    with rasterio.open(path, "w", driver="GTiff", height=side, width=side,
                       count=3, dtype="uint8", crs="EPSG:2056",
                       transform=from_origin(2662000, 1208000, GSD, GSD)) as dst:
        dst.write(data)
    tile = S.Tile(str(path), (2662000, 1207000, 2663000, 1208000), 2024)

    west = S.read_cell([tile], (2662000, 1207000, 2662500, 1207500), 64)
    east = S.read_cell([tile], (2662500, 1207000, 2663000, 1207500), 64)
    assert west.mean() == 0 and east.mean() == 255
