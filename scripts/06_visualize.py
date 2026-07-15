"""Точка входа: карта рекомендаций (folium) — точки с цветом по типу и обоснованием в popup."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import folium
import pandas as pd

from clinic_siting.utils.io import load_config, load_cells

# цвета и подписи по типу учреждения
_TYPE_COLOR = {"public": "#0F6E56", "private": "#993C1D"}
_TYPE_LABEL = {"public": "государственная", "private": "частная"}


def main():
    cfg = load_config()
    cells = load_cells(f"{cfg['output']['processed_dir']}/cells_estimated.parquet")
    recommended = pd.read_csv(f"{cfg['output']['tables_dir']}/recommendations.csv")

    min_lon, min_lat, max_lon, max_lat = cells.total_bounds
    fmap = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
                      zoom_start=13, tiles="cartodbpositron")

    # фон: клетки тонким контуром с подсказкой по данным
    folium.GeoJson(
        cells.to_json(),
        style_function=lambda f: {"color": "#bbb", "weight": 0.3, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["cell_id", "population", "existing_clinics", "wealth_score"],
            aliases=["Клетка", "Население", "Клиник рядом", "wealth_score"]),
    ).add_to(fmap)

    # рекомендованные точки: цвет по типу, размер по ценности, обоснование в popup
    for _, r in recommended.iterrows():
        color = _TYPE_COLOR[r["facility_type"]]
        popup = (f"<b>{r['cell_id']}</b><br>Тип: {_TYPE_LABEL[r['facility_type']]}"
                 f"<br>Ценность: {r['best_score']:.2f}")
        folium.CircleMarker(
            [r["center_lat"], r["center_lon"]], radius=8 + 6 * r["best_score"],
            color=color, fill=True, fill_color=color, fill_opacity=0.85, weight=2,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(fmap)

    n_public = int((recommended["facility_type"] == "public").sum())
    n_private = int((recommended["facility_type"] == "private").sum())
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
        padding:10px 14px;border:1px solid #999;border-radius:6px;font:13px sans-serif">
        <b>Рекомендации ({len(recommended)})</b><br>
        <span style="color:{_TYPE_COLOR['public']}">&#9679;</span> государственная ({n_public})<br>
        <span style="color:{_TYPE_COLOR['private']}">&#9679;</span> частная ({n_private})<br>
        <small>размер = ценность</small></div>"""
    fmap.get_root().html.add_child(folium.Element(legend))

    out_path = Path(cfg["output"]["maps_dir"]) / "recommendations.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print(f"Карта рекомендаций сохранена: {out_path}")


if __name__ == "__main__":
    main()
