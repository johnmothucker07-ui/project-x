# Clinic Siting — геоаналитика размещения медучреждений

Система рекомендует, **где** разместить медучреждение и **какого типа** (государственное / частное), по открытым геоданным. Гибрид: нейросеть оценивает район, классические алгоритмы решают, куда и что ставить.

## Принцип

- **Нейросеть (VLM)** — только суждения о районе (премиальность, тип застройки). Не считает.
- **Данные (OSM, Kontur)** — точные числа (население, конкуренты, дороги).
- **Алгоритмы (P-Median / MCLP)** — весь счёт и финальное решение.

## Пайплайн

```
grid → data (OSM + Kontur) → estimate (VLM) → optimize (scores + OSRM + solver) → validate
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # впиши QWEN_API_KEY
```

## OSRM (маршрутизация, локально в Docker)

Нужен для матрицы времени в пути. Один раз настроить:

```bash
# 1. скачать карту региона (пример: Центральный ФО)
wget https://download.geofabrik.de/russia/central-fed-district-latest.osm.pbf -P data/raw/osm/

# 2. подготовить (extract → partition → customize)
docker run -t -v "${PWD}/data/raw/osm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/central-fed-district-latest.osm.pbf
docker run -t -v "${PWD}/data/raw/osm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-partition /data/central-fed-district-latest.osrm
docker run -t -v "${PWD}/data/raw/osm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-customize /data/central-fed-district-latest.osrm

# 3. запустить сервер (порт 5000 — совпадает с config.yaml)
docker run -t -i -p 5000:5000 -v "${PWD}/data/raw/osm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld /data/central-fed-district-latest.osrm
```

Проверка: `curl "http://localhost:5000/route/v1/driving/37.6,55.75;37.62,55.76"` — должен вернуть JSON с маршрутом.

## Запуск пайплайна

```bash
python scripts/01_build_grid.py
python scripts/02_collect_data.py
python scripts/03_estimate.py
python scripts/04_optimize.py
python scripts/05_validate.py
# или всё разом:
python scripts/run_all.py
```

## Все параметры — в `config.yaml`

Границы района, размер клетки, модель, веса скоров, бюджет клиник, радиус 1000м — всё там. Меняешь → перезапускаешь. Для sensitivity-анализа меняешь только веса.

## Улучшения на будущее (архитектура готова)

- **Спутник**: `data/satellite.py` (заглушка есть), `estimate.use_satellite: true` в config
- **Уличные фото (Mapillary)**, **отзывы (2GIS)** — добавляются как новые модули в `data/`
- Роли компонентов при этом не меняются
