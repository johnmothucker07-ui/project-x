"""Визуальная проверка (п. 16).

Нужна людям, а не коду: глазами видно то, чего не ловит ни один тест —
сетка уехала, поликлиники легли в реку, допустимые клетки оказались
в промзоне. По п. 20 карта входит в приёмку этапа 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def city_map(grid, clinics, site_index, population, base_times, cfg,
             out_path: Path, scores: pd.DataFrame | None = None) -> Path:
    """HTML-карта города: население, поликлиники, площадки, время доступа."""
    import folium

    geographic = grid.to_crs(4326)
    centre = [float(geographic.geometry.centroid.y.mean()),
              float(geographic.geometry.centroid.x.mean())]
    fmap = folium.Map(location=centre, zoom_start=11, tiles="cartodbpositron")

    frame = geographic.copy()
    frame["population"] = population
    frame["time_min"] = base_times
    frame["covered"] = base_times <= cfg["T_minutes"]
    if scores is not None:
        frame = frame.merge(scores[["cell_id", "score"]], on="cell_id", how="left")

    folium.Choropleth(
        geo_data=frame.to_json(), data=frame, columns=["cell_id", "population"],
        key_on="feature.properties.cell_id", fill_color="YlOrRd",
        legend_name="население клетки", name="население").add_to(fmap)

    uncovered = folium.FeatureGroup(name=f"не покрыто за {cfg['T_minutes']} мин",
                                    show=True)
    for _, row in frame[~frame["covered"]].iterrows():
        folium.GeoJson(row.geometry,
                       style_function=lambda _f: {"color": "#b2182b",
                                                  "weight": 1, "fillOpacity": 0.25},
                       tooltip=f"{row.cell_id}: {row.time_min:.1f} мин, "
                               f"{row.population:.0f} чел.").add_to(uncovered)
    uncovered.add_to(fmap)

    sites = folium.FeatureGroup(name="допустимые площадки", show=False)
    for _, row in frame.iloc[site_index].iterrows():
        folium.CircleMarker([row.geometry.centroid.y, row.geometry.centroid.x],
                            radius=2, color="#2166ac", fill=True,
                            tooltip=row.cell_id).add_to(sites)
    sites.add_to(fmap)

    existing = folium.FeatureGroup(name="существующие поликлиники", show=True)
    for _, row in clinics.to_crs(4326).iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=4,
                            color="#1a9850", fill=True,
                            tooltip=str(row.get("name", ""))).add_to(existing)
    existing.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def solutions_map(grid, clinics, solutions: dict[str, tuple], site_index,
                  out_path: Path) -> Path:
    """Где каждый вариант поставил точки — главная картинка для отчёта."""
    import folium

    geographic = grid.to_crs(4326)
    centre = [float(geographic.geometry.centroid.y.mean()),
              float(geographic.geometry.centroid.x.mean())]
    fmap = folium.Map(location=centre, zoom_start=11, tiles="cartodbpositron")

    for _, row in clinics.to_crs(4326).iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=3,
                            color="#999999", fill=True, opacity=0.5).add_to(fmap)

    palette = {"B": "#1f78b4", "M": "#33a02c", "C": "#e31a1c",
               "D": "#ff7f00", "E": "#6a3d9a"}
    for name, sites in solutions.items():
        if not sites:
            continue
        group = folium.FeatureGroup(name=f"вариант {name}", show=True)
        for position in sites:
            cell = geographic.iloc[site_index[position]]
            folium.Marker(
                [cell.geometry.centroid.y, cell.geometry.centroid.x],
                tooltip=f"{name}: {cell.cell_id}",
                icon=folium.Icon(color="blue", icon="plus")).add_to(group)
            folium.Circle([cell.geometry.centroid.y, cell.geometry.centroid.x],
                          radius=1125, color=palette.get(name, "#333333"),
                          fill=False, weight=2).add_to(group)
        group.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def cell_inspector(features: pd.DataFrame, texts: pd.Series,
                   scores: pd.DataFrame | None, out_path: Path,
                   sample: int = 20, seed: int = 0) -> Path:
    """Инспектор клетки: текст, скор и обоснование рядом (п. 16).

    Берутся крайние клетки по скору — на них видно, различает ли модель
    что-нибудь вообще, или ставит всем одно и то же."""
    rows = features.copy()
    rows["text"] = texts.to_numpy()
    if scores is not None:
        rows = rows.merge(scores, on="cell_id", how="left")
        ordered = rows.sort_values("score")
        chosen = pd.concat([ordered.head(sample // 2), ordered.tail(sample // 2)])
    else:
        chosen = rows.sample(n=min(sample, len(rows)), random_state=seed)

    blocks = []
    for _, row in chosen.iterrows():
        score = row.get("score")
        blocks.append(
            f"<div class='cell'><h3>{row.cell_id}"
            + (f" — скор {score:.0f}</h3>" if pd.notna(score) else "</h3>")
            + (f"<p class='why'>{row.get('rationale', '')}</p>"
               if scores is not None else "")
            + f"<p class='text'>{row.text}</p></div>")

    html = ("<html><head><meta charset='utf-8'><style>"
            "body{font-family:system-ui;margin:2rem;max-width:60rem}"
            ".cell{border-bottom:1px solid #ddd;padding:1rem 0}"
            ".why{color:#555;font-style:italic}"
            ".text{color:#222}</style></head><body>"
            "<h1>Инспектор клеток</h1>"
            "<p>Крайние по скору клетки: вверху те, которым модель поставила "
            "меньше всего, внизу — больше всего.</p>"
            + "".join(blocks) + "</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def clinic_review_queue(medical, state_clinics, out_path: Path) -> Path:
    """Очередь ручной проверки поликлиник (п. 4.5, Л9).

    Человеку надо решить по каждому объекту, государственная ли это взрослая
    поликлиника. По тегам OSM это неразрешимо: operator заполнен у 6 %."""
    chosen = set(state_clinics["osm_id"]) if "osm_id" in state_clinics else set()
    queue = medical.copy()
    queue["selected_by_name"] = queue["osm_id"].isin(chosen) if "osm_id" in queue else False
    queue["is_state"] = ""         # заполняет человек
    queue["is_adult_primary"] = ""
    queue["comment"] = ""
    columns = [c for c in ("osm_id", "name", "amenity", "healthcare",
                           "healthcare_speciality", "operator", "operator_type",
                           "selected_by_name", "is_state", "is_adult_primary",
                           "comment") if c in queue.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    queue[columns].to_csv(out_path, index=False, encoding="utf-8")
    return out_path
