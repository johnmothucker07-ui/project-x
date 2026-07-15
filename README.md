# Clinic Siting — геоаналитика размещения медучреждений

Система рекомендует, **где** разместить медучреждение и **какого типа** (государственное / частное), по открытым геоданным. Гибрид: нейросеть оценивает район, классические алгоритмы решают, куда и что ставить.

## Принцип (разделение ролей)

- **Нейросеть (VLM)** — только *суждения* о районе (премиальность, тип застройки). Ничего не считает.
- **Данные (OSM, Kontur)** — точные *числа* (население, конкуренты).
- **Алгоритмы (P-Median / MCLP на PuLP)** — весь *счёт* и финальное решение (точки + тип).

## Пайплайн

```
grid → data (OSM + Kontur) → estimate (VLM) → optimize (scores + solver) → validate
```

---

## Как воспроизвести решение

Инструкция для сокомандников — как поднять проект с нуля.

### 1. Окружение (Python 3.11)

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux/Mac:             source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ключ API модели (SiliconFlow)

Проект использует OpenAI-совместимый провайдер **SiliconFlow** (эндпоинт и модель — в `config.yaml`, раздел `estimate`).

1. Получи API-ключ на https://siliconflow.com (раздел API Keys).
2. Создай `.env` из шаблона и впиши ключ:

```bash
cp .env.example .env      # Windows: copy .env.example .env
# в .env:  QWEN_API_KEY=sk-...   (ключ SiliconFlow; имя переменной оставь как есть)
```

`.env` в `.gitignore` — ключ в репозиторий не попадёт. Сменить провайдера/модель можно одной строкой в `config.yaml` (код провайдер-независимый).

### 3. Данные о населении (Kontur, ~174 МБ, вручную)

Файл большой, в git не хранится — скачай отдельно:

1. Открой https://data.humdata.org/dataset/kontur-population-russian-federation
2. Скачай GeoPackage `kontur_population_RU_YYYYMMDD.gpkg.gz`, распакуй `.gz`.
3. Положи по пути (имя ровно такое, как в `config.yaml`):
   `data/raw/population/kontur_population.gpkg`

Клиники из OSM (Overpass) и кэш модели скачиваются/считаются автоматически при прогоне.

### 4. Прогон пайплайна

```bash
python scripts/01_build_grid.py       # сетка клеток по bbox из config → data/interim
python scripts/02_collect_data.py     # OSM (клиники, конкуренты, выжимка) + население → data/processed/cells.parquet
python scripts/03_estimate.py         # VLM: wealth_score + area_character по клеткам (кэш в data/cache)
python scripts/04_optimize.py         # два скора + MILP → outputs/tables/recommendations.csv
python scripts/05_validate.py         # доступность до/после + baseline → outputs/tables/validation.csv
python scripts/06_visualize.py        # карта рекомендаций → outputs/maps/recommendations.html
python scripts/07_report_figures.py   # графики для отчёта → outputs/figures/
# или всё разом:
python scripts/run_all.py
```

Пробный прогон Estimate на N клетках (экономит токены): `python scripts/03_estimate.py 5`.
Повторный запуск `03` берёт готовые клетки из кэша — не тратит токены.

### Результат

- `data/processed/cells.parquet` — таблица клеток (население, конкуренты, контекст).
- `data/processed/cells_estimated.parquet` — + оценки модели (wealth_score, area_character).
- `outputs/tables/recommendations.csv` — итог: 8 точек с типом (public/private) и ценностью.

---

## Пример результата (тестовый район — центр Москвы)

На bbox из config (кусок центра Москвы, сетка 500 м → 182 клетки, 439 клиник-конкурентов из OSM):

**Рекомендации (`outputs/tables/recommendations.csv`):** 8 точек — **6 частных + 2 государственные**, минимальное расстояние между ними 1412 м (норматив ≥ 1000 м соблюдён).

**Валидация доступности (`outputs/tables/validation.csv`)** — среднее по населению время до ближайшей клиники, мин:

| Сценарий | before | after | improvement |
|---|---|---|---|
| Модель | 0.480 | **0.450** | **0.030** |
| Baseline (по плотности) | 0.480 | 0.462 | 0.018 |

→ после добавления точек стало ближе (**after < before**), и модель улучшает доступность **в 1.65× сильнее** наивного baseline.
Абсолютные значения малы — центр Москвы уже насыщен клиниками; в недообслуженном районе эффект был бы заметнее.

**Визуализация:** `outputs/maps/recommendations.html` (карта: точки с цветом по типу и обоснованием),
`outputs/figures/` (графики валидации, wealth_score, распределения скоров).

## Все параметры — в `config.yaml`

Границы района (`bbox`), размер клетки, модель/эндпоинт, веса скоров, бюджет клиник, радиус 1000 м — всё там. Меняешь → перезапускаешь. Для sensitivity-анализа меняешь только веса.

## Известные упрощения текущего MVP

Делаем простой работающий продукт; улучшения добавляются модульно, не переписывая архитектуру.

- **Доступность считается по прямой (haversine), без OSRM.** Реальное время по дорогам (OSRM в Docker) — улучшение (сигнатуры в `optimize/routing.py` уже готовы).
- **`area_character` выходит однородным** (контекст беден) — фильтр промзоны пока почти не работает. Улучшение: обогатить контекст (landuse) или брать тип из OSM landuse.
- **`wealth_score` — прокси** (текстовый контекст) — усилится спутником/уличными фото.
- Конкуренты чистятся от лабораторий/косметологии (`exclude_*` в config); точную классификацию дал бы рубрикатор 2ГИС.

Полный список улучшений — в `TODO.md` (раздел «ОТЛОЖЕНО»), замысел и обоснования — в `project_final.md`.

## Улучшения на будущее (архитектура готова)

- **Спутник**: `data/satellite.py` + `estimate.use_satellite: true` в config → картинка в тот же вызов VLM.
- **OSRM**: реальное время в пути вместо haversine.
- **Уличные фото (Mapillary)**, **отзывы/рубрики (2GIS)** — новые модули в `data/`.
- Роли компонентов при этом не меняются.
