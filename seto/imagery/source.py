"""Снимок под клетку: индекс плиток и чтение окна (Ш7, шаг 4.2).

Плитки swisstopo — Cloud-Optimized GeoTIFF, поэтому окно читается прямо по
сети: одна плитка 1 км² при 0,1 м весит 76 МБ, а нам из неё нужен квадрат
500 м. Скачивать целиком незачем, и это заодно снимает вопрос о нагрузке на
их инфраструктуру.

Контракт: снимок покрывает РОВНО тот квадрат, по которому считаются признаки
OSM и население. Ключ связи — cell_id. Если геометрии разъедутся, вариант D
будет смотреть не на ту клетку, и разница D − C перестанет что-либо значить.
Поэтому окно задаётся границами клетки, а не центром с запасом, и частичное
покрытие возвращает None, а не чёрные поля.

Сетка строится в той же проекции, что и снимок (для Швейцарии EPSG:2056),
поэтому клетка — осевой квадрат в системе растра и окно читается без
поворота. Это решение записано в конфиге города.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Чтобы GDAL не перебирал каталог на каждом открытии по сети.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

_USER_AGENT = "seto-research/1.0 (university project)"


class CoverageError(RuntimeError):
    """Клетка не покрыта снимком целиком."""


@dataclass(frozen=True)
class Tile:
    """Плитка снимка: где лежит и что покрывает (в CRS растра)."""
    href: str
    bounds: tuple[float, float, float, float]
    year: int

    def covers(self, other: tuple[float, float, float, float]) -> bool:
        l, b, r, t = self.bounds
        ol, ob, orr, ot = other
        return not (orr <= l or ol >= r or ot <= b or ob >= t)

    @property
    def path(self) -> str:
        """Чем открывать: по сети через vsicurl, с диска — как есть."""
        if self.href.startswith(("http://", "https://")):
            return f"/vsicurl/{self.href}"
        return self.href


def index_tiles(stac_url: str, bbox_wgs84: tuple[float, float, float, float],
                gsd: float = 0.1, year: int | None = None) -> list[Tile]:
    """Плитки на рамку, по одной на каждую позицию.

    В каталоге одна и та же позиция есть за несколько лет съёмки. Брать
    вперемешку нельзя: разные годы — разный сезон, освещение и состояние
    застройки, и это попало бы в сравнение разрешений как посторонний
    источник различий.

    `year` задаёт единый год. Позиции, которых в этом году нет, просто не
    попадают в индекс — клетки на них потом отсеются по покрытию. Это лучше,
    чем подставить соседний год и получить шов посреди города. Без `year`
    берётся самый свежий снимок на каждую позицию — годится для просмотра,
    но не для сравнения."""
    marker = f"_{gsd}_"
    by_position: dict[tuple, Tile] = {}
    url = (f"{stac_url}/items?bbox={','.join(str(v) for v in bbox_wgs84)}"
           f"&limit=100")

    while url:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)

        for feature in page.get("features", []):
            href = next((asset["href"] for name, asset in feature["assets"].items()
                         if marker in name), None)
            if href is None:
                continue
            # id вида swissimage-dop10_2021_2662-1207
            _, flown, position = feature["id"].rsplit("_", 2)
            if year is not None and int(flown) != year:
                continue
            easting, northing = (int(v) * 1000 for v in position.split("-"))
            tile = Tile(href=href, year=int(flown),
                        bounds=(easting, northing, easting + 1000, northing + 1000))
            known = by_position.get((easting, northing))
            if known is None or tile.year > known.year:
                by_position[(easting, northing)] = tile

        url = next((link["href"] for link in page.get("links", [])
                    if link.get("rel") == "next"), None)

    return sorted(by_position.values(), key=lambda t: t.bounds)


def read_cell(tiles: list[Tile], bounds: tuple[float, float, float, float],
              out_px: int) -> np.ndarray:
    """Прочитать квадрат клетки как RGB-массив (out_px, out_px, 3).

    Клетка 500 м и плитка 1 км выровнены по общей сетке, поэтому обычно
    клетка целиком лежит внутри одной плитки. Склейка на случай, если это
    перестанет быть так: молча обрезать клетку по краю плитки нельзя."""
    left, bottom, right, top = bounds
    overlapping = [t for t in tiles if t.covers(bounds)]
    if not overlapping:
        raise CoverageError(f"нет плиток на клетку {bounds}")

    canvas = np.zeros((out_px, out_px, 3), dtype=np.uint8)
    filled = np.zeros((out_px, out_px), dtype=bool)
    metres_per_px = (right - left) / out_px

    with rasterio.Env(**GDAL_ENV):
        for tile in overlapping:
            tl, tb, tr, tt = tile.bounds
            # пересечение клетки с плиткой
            il, ib = max(left, tl), max(bottom, tb)
            ir, it = min(right, tr), min(top, tt)
            if ir <= il or it <= ib:
                continue

            # куда положить кусок: y растёт вниз, поэтому отсчёт от верха
            col = int(round((il - left) / metres_per_px))
            row = int(round((top - it) / metres_per_px))
            width = int(round((ir - il) / metres_per_px))
            height = int(round((it - ib) / metres_per_px))
            if width <= 0 or height <= 0:
                continue

            with rasterio.open(tile.path) as src:
                window = from_bounds(il, ib, ir, it, src.transform)
                piece = src.read([1, 2, 3], window=window,
                                 out_shape=(3, height, width),
                                 boundless=True, fill_value=0)
            canvas[row:row + height, col:col + width] = np.moveaxis(piece, 0, -1)
            filled[row:row + height, col:col + width] = True

    if not filled.all():
        raise CoverageError(
            f"клетка {bounds} покрыта снимком на "
            f"{100 * filled.mean():.1f} % — пропускаем, чёрные поля исказили бы оценку")
    return canvas
