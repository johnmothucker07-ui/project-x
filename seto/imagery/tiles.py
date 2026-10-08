"""Спутниковые снимки по геометрии клетки (п. 5, Ш7).

Контракт жёсткий: изображение должно покрывать РОВНО ту территорию, по которой
считаются признаки OSM, текст варианта C и население. Ключ связи — cell_id.
Иначе разница D − C перестанет означать вклад изображения: модель будет
смотреть не на ту клетку.

ПРОВАЙДЕР НЕ ВЫБРАН. Это блокировка Л6: нужно подтвердить источник и прочитать
его условия использования — разрешены ли автоматическая загрузка и обработка
моделью. Подставлять произвольный тайл-сервер нельзя: у большинства это прямо
запрещено лицензией, и прогон стал бы юридически негодным.

Вся механика ниже написана и покрыта тестами; не хватает только строки
`imagery.provider` в конфиге и ключа.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EARTH_CIRCUMFERENCE_M = 40_075_016.686


class ProviderNotConfigured(RuntimeError):
    """Провайдер снимков не выбран — это Л6, а не ошибка кода."""


@dataclass
class TileSpec:
    """Что именно качать для одной клетки."""
    cell_id: str
    zoom: int
    tiles: list[tuple[int, int]]       # (x, y) тайлов XYZ
    bbox_wgs84: tuple[float, float, float, float]
    meters_per_pixel: float


def resolution_at(latitude: float, zoom: int, tile_px: int = 256) -> float:
    """Метров на пиксель в Web Mercator.

    Зависит от широты: один и тот же зум в Москве и в Сочи даёт разный
    масштаб, поэтому зум подбирается под целевое разрешение, а не задаётся
    числом (п. 5.2)."""
    return (EARTH_CIRCUMFERENCE_M * math.cos(math.radians(latitude))
            / (tile_px * 2 ** zoom))


def zoom_for(latitude: float, target_m_per_px: float, tile_px: int = 256,
             max_zoom: int = 22) -> int:
    """Наименьший зум, дающий разрешение не хуже целевого."""
    if target_m_per_px <= 0:
        raise ValueError("целевое разрешение должно быть больше нуля")
    for zoom in range(max_zoom + 1):
        if resolution_at(latitude, zoom, tile_px) <= target_m_per_px:
            return zoom
    return max_zoom


def tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Номер тайла XYZ, в котором лежит точка."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    radians = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tiles_for_bbox(bbox: tuple[float, float, float, float],
                   zoom: int) -> list[tuple[int, int]]:
    """Все тайлы, покрывающие рамку."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = tile_xy(min_lon, max_lat, zoom)      # верхний левый
    x1, y1 = tile_xy(max_lon, min_lat, zoom)      # нижний правый
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def plan_cell(cell_id: str, bbox_wgs84: tuple[float, float, float, float],
              cfg: dict) -> TileSpec:
    """Что качать для одной клетки, с запасом контекста из конфига."""
    imagery = cfg["imagery"]
    margin_m = imagery.get("context_margin_m", 0)
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    if margin_m:
        lat_pad = margin_m / 111_320.0
        lon_pad = lat_pad / max(0.01, math.cos(math.radians((min_lat + max_lat) / 2)))
        min_lon, max_lon = min_lon - lon_pad, max_lon + lon_pad
        min_lat, max_lat = min_lat - lat_pad, max_lat + lat_pad

    centre_lat = (min_lat + max_lat) / 2
    zoom = zoom_for(centre_lat, imagery["target_m_per_px"])
    bbox = (min_lon, min_lat, max_lon, max_lat)
    return TileSpec(cell_id=cell_id, zoom=zoom, tiles=tiles_for_bbox(bbox, zoom),
                    bbox_wgs84=bbox,
                    meters_per_pixel=resolution_at(centre_lat, zoom))


def plan_all(grid, cfg: dict) -> list[TileSpec]:
    """План загрузки по всем клеткам. Зум один на город (п. 5.1)."""
    geographic = grid.to_crs(4326)
    specs = [plan_cell(row.cell_id, tuple(geom.bounds), cfg)
             for row, geom in zip(grid.itertuples(), geographic.geometry)]
    zooms = {spec.zoom for spec in specs}
    if len(zooms) > 1:
        # один уровень детализации на все клетки и все города (п. 5.1)
        best = max(zooms)
        specs = [TileSpec(s.cell_id, best, tiles_for_bbox(s.bbox_wgs84, best),
                          s.bbox_wgs84, s.meters_per_pixel) for s in specs]
    return specs


def quality_flags(image: np.ndarray) -> dict:
    """Автопроверки снимка (п. 5.4): пустой, однотонный, много no-data."""
    if image.size == 0:
        return {"image_qc": "empty", "ok": False}
    flat = image.reshape(-1, image.shape[-1]) if image.ndim == 3 else image.reshape(-1, 1)
    unique = len(np.unique(flat[:: max(1, len(flat) // 1000)], axis=0))
    nodata_share = float((flat == 0).all(axis=1).mean()) if flat.ndim == 2 else 0.0
    if unique <= 2:
        return {"image_qc": "flat", "ok": False, "unique_colors": unique}
    if nodata_share > 0.5:
        return {"image_qc": "mostly_nodata", "ok": False,
                "nodata_share": nodata_share}
    return {"image_qc": "ok", "ok": True, "unique_colors": unique,
            "nodata_share": nodata_share}


def require_provider(cfg: dict) -> str:
    """Проверка, что провайдер выбран. Без него загрузка не начинается."""
    provider = (cfg.get("imagery") or {}).get("provider")
    if not provider:
        raise ProviderNotConfigured(
            "провайдер снимков не выбран (imagery.provider). Это Л6: нужно "
            "подтвердить источник и прочитать его условия использования — "
            "разрешены ли автоматическая загрузка и обработка моделью. "
            "Механика загрузки готова, не хватает только этого решения.")
    return provider
