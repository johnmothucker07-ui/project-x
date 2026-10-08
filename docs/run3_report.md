# Отчёт по городу: —

## Данные

- граница: 928.7 км², источник `union_of_level`
- клеток: 3958 по 500 м
- население: 12,103,096, расхождение при переносе 0.000 %
- поликлиник в базовой сети: 541 (по названию — реестра нет (Л7))
- граф: 1,322,806 узлов, выброшено 22,269
- непривязанных клеток: 24, недостижимых: 67
- допустимых площадок: 1497 из 3958
- доля зданий с этажностью: 51.9 %

## Главная таблица (T = 15, k = 2)

| variant | Cov_15 | dCov_15 | dN_15 | worst_decile_min | mean_time_min | share_below_floor_pct | p10 | p25 |
|---|---|---|---|---|---|---|---|---|
| исходная сеть | 68.34 | 0.00 | 0.00 | 30.66 | 13.10 | 49.29 | 0.00 | 0.00 |
| B | 68.82 | 0.48 | 57 897.33 | 30.20 | 12.99 | 48.99 | 0.00 | 0.00 |
| M | 69.06 | 0.72 | 86 821.95 | 30.66 | 13.03 | 48.95 | 0.00 | 0.00 |
| C | 68.92 | 0.58 | 70 243.33 | 30.39 | 12.99 | 49.00 | 0.00 | 0.00 |

## Разницы между вариантами

| pair | dCov_15_pp | dN_15_people | above_A_p95 | percentile_in_A |
|---|---|---|---|---|
| B − M | -0.24 | -28 924.61 | — | — |
| B − C | -0.10 | -12 346.00 | — | — |
| C − M | -0.14 | -16 578.61 | — | — |
| B − A (медиана) | 0.37 | — | да | 99.70 |
| M − A (медиана) | 0.61 | — | да | 100.00 |
| C − A (медиана) | 0.47 | — | да | 100.00 |

## Устойчивость по порогу

| variant | Cov_10 | dCov_10 | dN_10 | Cov_15 | dCov_15 | dN_15 | Cov_20 | dCov_20 | dN_20 |
|---|---|---|---|---|---|---|---|---|---|
| исходная сеть | 40.53 | 0.00 | 0.00 | 68.34 | 0.00 | 0.00 | 84.70 | 0.00 | 0.00 |
| B | 40.84 | 0.31 | 37 475.23 | 68.82 | 0.48 | 57 897.33 | 85.10 | 0.41 | 49 148.21 |
| M | 40.94 | 0.41 | 50 098.42 | 69.06 | 0.72 | 86 821.95 | 84.88 | 0.19 | 22 402.89 |
| C | 40.87 | 0.34 | 41 225.74 | 68.92 | 0.58 | 70 243.33 | 85.08 | 0.38 | 46 327.64 |

## Критерии пилота

```json
{
  "quality": {
    "metric": "dCov_15",
    "A_p95": 0.28382431792292306,
    "C_dCov": 0.58037492122142,
    "B_dCov": 0.4783679492175992,
    "M_dCov": 0.7173532092074026,
    "status": "нет данных: вариант D не считался (нет снимков, Л6)"
  },
  "costs": {
    "mode": "onboard_city",
    "total_seconds": 372.58,
    "by_tag": {
      "shared_prep": 247.8,
      "variant:A": 0.05,
      "variant:B": 34.75,
      "variant:C": 40.17,
      "variant:M": 49.8
    },
    "stages": [
      {
        "stage": "boundary",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 20.16
      },
      {
        "stage": "grid",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 3.43,
        "cells": 3958,
        "fully_inside": 3492,
        "on_border": 466,
        "cell_size_m": 500.0
      },
      {
        "stage": "osm_layers",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 59.42,
        "buildings_total": 195803,
        "buildings_with_type": 109358,
        "buildings_with_levels": 101699,
        "buildings_levels_share": 0.5194,
        "roads_total": 436219,
        "roads_by_class": {
          "footway": 207917,
          "service": 136947,
          "path": 18275,
          "steps": 17338,
          "tertiary": 12536,
          "secondary": 12476,
          "residential": 9642,
          "unclassified": 5611,
          "primary": 4909,
          "secondary_link": 1854,
          "cycleway": 1788,
          "track": 1139
        },
        "points_total": 177455
      },
      {
        "stage": "population",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 15.18,
        "source_total": 12103095.7,
        "transferred_total": 12103095.7,
        "loss_share": 0.0,
        "pixels_used": 83498,
        "nodata_pixels": 0,
        "partial_pixels": 1827,
        "outside_grid": 54595.0
      },
      {
        "stage": "features",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 66.07
      },
      {
        "stage": "clinics",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 0.03,
        "candidates": 3298,
        "state_proxy": 541
      },
      {
        "stage": "graph",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 17.38,
        "nodes": 1322806,
        "edges": 1572730,
        "dropped_nodes": 22269
      },
      {
        "stage": "times_existing",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 8.37,
        "unattached_cells": 24,
        "unreachable_cells": 67,
        "unreachable_people": 41970.680542192116
      },
      {
        "stage": "sites",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 28.81,
        "feasible": 1497,
        "total_cells": 3958,
        "excluded_by": {
          "мало жилой и общественной застройки": 1885,
          "вода": 1,
          "парк или зелень": 19,
          "промзона": 34,
          "железная дорога": 1,
          "кладбище": 0,
          "военная территория": 0,
          "не привязана к графу": 0,
          "ближе минимума к существующей поликлинике": 521
        }
      },
      {
        "stage": "times_sites",
        "tag": "shared_prep",
        "mode": "onboard_city",
        "seconds": 28.96,
        "matrix": [
          1497,
          3958
        ],
        "max_feasible_k": 2
      },
      {
        "stage": "scoring",
        "tag": "variant:C",
        "mode": "onboard_city",
        "seconds": 1.4,
        "cells": 3958,
        "from_cache": 3958,
        "api_calls": 0,
        "retries": 0,
        "imputed": 0,
        "imputed_share": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "failures": []
      },
      {
        "stage": "solve",
        "tag": "variant:B",
        "mode": "onboard_city",
        "seconds": 34.75
      },
      {
        "stage": "solve",
        "tag": "variant:M",
        "mode": "onboard_city",
        "seconds": 49.8
      },
      {
        "stage": "solve",
        "tag": "variant:C",
        "mode": "onboard_city",
        "seconds": 38.81
      },
      {
        "stage": "solve",
        "tag": "variant:A",
        "mode": "onboard_city",
        "seconds": 0.05
      }
    ]
  },
  "reliability": {
    "statuses": {
      "B": "ok",
      "M": "ok",
      "C": "ok",
      "D": "missing_external",
      "A": "ok"
    },
    "manual_fixes": 0,
    "status": "выполнен"
  }
}
```

## Статусы вариантов

- **B**: ok
- **M**: ok
- **C**: ok — корреляция весов с B 0.9795
- **D**: missing_external — нет скоров режима image
- **A**: ok — 1000 наборов
