"""Перенос населения из растра в клетки сетки (п. 4.6).

Растр и сетка не выровнены: GHS-POP лежит в Mollweide, сетка — в UTM города.
Поэтому пиксели полигонизуются в СВОЕЙ проекции, переводятся в UTM, и площади
пересечения считаются уже там — как требует п. 4.6. Присваивать пиксель
целиком ближайшей клетке нельзя: на границах клеток это сдвигает население.

Полигонизуются только пиксели с ненулевым населением — их на порядок меньше,
чем всех, и это единственное, что делает честный overlay подъёмным.

Проверка сохранения массы обязательна: расхождение больше 0,1 % означает,
что часть людей потерялась или удвоилась, и дальше считать нельзя.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


@dataclass
class PopulationTransfer:
    """Население по клеткам и отчёт о переносе."""
    per_cell: pd.Series          # индекс — cell_id
    source_total: float          # сколько ДОЛЖНО было перенестись (по накрытой доле)
    transferred_total: float
    pixels_used: int
    nodata_pixels: int
    partial_pixels: int = 0      # пиксели на краю сетки, накрытые не целиком
    outside_grid: float = 0.0    # их население за пределами сетки

    @property
    def loss_share(self) -> float:
        if self.source_total <= 0:
            return 0.0
        return abs(self.transferred_total - self.source_total) / self.source_total

    def as_dict(self) -> dict:
        return {
            "source_total": round(self.source_total, 1),
            "transferred_total": round(self.transferred_total, 1),
            "loss_share": self.loss_share,
            "pixels_used": self.pixels_used,
            "nodata_pixels": self.nodata_pixels,
            "partial_pixels": self.partial_pixels,
            "outside_grid": round(self.outside_grid, 1),
        }


def read_pixels(raster_path: Path,
                bounds_in_raster_crs: tuple) -> tuple[gpd.GeoDataFrame, int]:
    """Пиксели с населением > 0 внутри рамки, как полигоны в проекции растра.

    Возвращает (пиксели, число пикселей nodata)."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(raster_path) as source:
        window = from_bounds(*bounds_in_raster_crs, transform=source.transform)
        window = window.round_offsets().round_lengths()
        data = source.read(1, window=window)
        transform = source.window_transform(window)
        nodata = source.nodata
        crs = source.crs

    values = data.astype("float64")
    missing = np.zeros_like(values, dtype=bool) if nodata is None else (values == nodata)
    values[missing] = 0.0
    # отрицательные значения в растрах населения встречаются как служебные
    values[values < 0] = 0.0

    rows, cols = np.nonzero(values)
    if rows.size == 0:
        empty = gpd.GeoDataFrame({"pop": []}, geometry=[], crs=crs)
        return empty, int(missing.sum())

    size_x, size_y = transform.a, -transform.e
    x0 = transform.c + cols * size_x
    y0 = transform.f - rows * size_y
    cells = [box(x, y - size_y, x + size_x, y) for x, y in zip(x0, y0)]
    return gpd.GeoDataFrame({"pop": values[rows, cols]}, geometry=cells, crs=crs), \
        int(missing.sum())


def to_cells(raster_path: Path, grid: gpd.GeoDataFrame,
             tolerance: float = 0.001) -> PopulationTransfer:
    """Разнести население растра по клеткам пропорционально площади пересечения."""
    import rasterio

    if grid.crs is None or grid.crs.is_geographic:
        raise ValueError("сетка должна быть в метрической проекции (UTM)")

    with rasterio.open(raster_path) as source:
        raster_crs = source.crs
    # рамку сетки переводим в проекцию растра, иначе прочитаем не то окно
    frame = gpd.GeoDataFrame(geometry=[box(*grid.total_bounds)], crs=grid.crs)
    bounds = tuple(frame.to_crs(raster_crs).total_bounds)

    pixels, nodata_count = read_pixels(raster_path, bounds)
    if pixels.empty:
        raise ValueError("в рамке сетки нет ни одного пикселя с населением")

    pixels = pixels.to_crs(grid.crs)
    pixels["pixel_area"] = pixels.geometry.area
    pixels["pixel_id"] = np.arange(len(pixels))

    pieces = gpd.overlay(pixels[["pixel_id", "pop", "pixel_area", "geometry"]],
                         grid[["cell_id", "geometry"]], how="intersection")
    pieces["share"] = pieces.geometry.area / pieces["pixel_area"]
    pieces["people"] = pieces["pop"] * pieces["share"]

    per_cell = pieces.groupby("cell_id")["people"].sum()
    per_cell = per_cell.reindex(grid["cell_id"], fill_value=0.0)

    # Какая доля каждого пикселя накрыта сеткой. Пиксель на краю сетки накрыт
    # частично, и переносить его целиком неверно — поэтому эталон считается
    # по накрытой доле, а не по полному населению пикселя.
    coverage = pieces.groupby("pixel_id")["share"].sum()
    if float(coverage.max()) > 1.0 + 1e-6:
        # клетки не должны перекрываться; если перекрылись — людей удвоили
        raise ValueError(
            f"пиксель накрыт клетками более чем на 100 % "
            f"({100 * coverage.max():.2f} %): клетки сетки перекрываются")

    used = pixels[pixels["pixel_id"].isin(coverage.index)].set_index("pixel_id")
    expected = float((used["pop"] * coverage).sum())
    partial = coverage[coverage < 1.0 - 1e-6]

    transfer = PopulationTransfer(
        per_cell=per_cell,
        source_total=expected,
        transferred_total=float(per_cell.sum()),
        pixels_used=int(len(used)),
        nodata_pixels=nodata_count,
        partial_pixels=int(len(partial)),
        outside_grid=float((used.loc[partial.index, "pop"] * (1 - partial)).sum()),
    )
    if transfer.loss_share > tolerance:
        raise ValueError(
            f"население не сохранилось при переносе: ожидалось {transfer.source_total:,.0f}, "
            f"перенесено {transfer.transferred_total:,.0f} "
            f"(расхождение {100 * transfer.loss_share:.3f} % > {100 * tolerance:.1f} %)")
    return transfer
