# Протокол эксперимента (предрегистрация)

Заморожен: 2026-10-08T23:59:56.814343+00:00
Коммит: `7d7bb93d433e02bb68127a7786556272a6744b92`
Модель: `google/gemma-4-31B-it`
Срез данных: 2026-10-08
Хеш протокола: `722c59ff427963ea098e424cf902d58d768bc8e6b515b8333cc59a39e187e532`

## Города

- **moscow** — dev

## Параметры

```json
{
  "scenario": "state",
  "k": 2,
  "k_sensitivity": [
    1
  ],
  "T_minutes": 15,
  "T_sensitivity": [
    10,
    20
  ],
  "walk_speed_m_per_min": 75,
  "t_max_minutes": 60,
  "snap_max_m": 300,
  "min_dist_to_existing_m": 500,
  "min_dist_between_new_m": 500,
  "alpha": 1.0,
  "alpha_sensitivity": [
    0.5,
    2.0
  ],
  "random_draws": 1000,
  "seed": 0,
  "imputation_max_share": 0.05,
  "feasible_site": {
    "min_built_area_share": 0.05,
    "exclude_landuse": [
      "industrial",
      "railway",
      "cemetery",
      "military"
    ],
    "exclude_natural": [
      "water"
    ],
    "exclude_leisure": [
      "park",
      "garden"
    ]
  },
  "grid": {
    "size_m": 500,
    "origin_rule": "floor_to_size",
    "id_format": "cell_{:06d}"
  }
}
```

## Метрики

- Cov_T
- dCov_T
- dN_T
- worst_decile
- mean_time
- 2SFCA: share_below_floor, p10, p25

## Критерии пилота

- **quality**: dCov_T варианта D выше 95-го процентиля варианта A
- **costs**: активное время человека у D не более половины от E
- **reliability**: D дал допустимое решение без ручной правки

## Чего эксперимент НЕ делает

- порог допустимой потери δ
- порог ε
- среднее A_i по 2SFCA

После просмотра результатов проверки ничего из этого не меняется.
