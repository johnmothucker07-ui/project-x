"""
ЭТАП 2-alt · Аналитический baseline: wealth_score БЕЗ модели.

Зачем: сейчас wealth_score от VLM — фактически счётчик клиник (Спирмен 0.93).
Чтобы честно измерить, что вообще даёт модель, нужен конкурент, считающий
достаток теми же открытыми данными, но кодом. Этот модуль — он.

Подставляется в ТОТ ЖЕ слот пайплайна, что и estimate.py: отдаёт таблицу
[cell_id, wealth_score, ...], дальше Optimize и Validate не меняются.

ВАЖНО, почему доли, а не счётчики: счётчики объектов меряют плотность городской
ткани, а не достаток (плотность окружения ↔ число клиник = 0.63). Клиники стоят
там же, где кафе и магазины. Доля внутри категории от плотности не зависит.

ZERO-LABEL: веса в config выставлены руками и не подбирались по ценам м².
Это принципиально — арм сравнивается с zero-shot VLM, который разметки не видел.
Обучаемый вариант (регрессия на ценах) — отдельный арм, не этот файл.
"""
from __future__ import annotations

import pandas as pd

# шкала VLM (0-10) — держим ту же, чтобы скор подставлялся в слот без правок
_SCORE_MAX = 10


def _group_of(column: str) -> str:
    """env_shop_clothes -> shop. Группа нужна, чтобы считать долю внутри своей категории."""
    return column.removeprefix("env_").split("_", 1)[0]


def composition_shares(features):
    """Доли объектов внутри своей категории (состав района вместо плотности).

    Возвращает DataFrame [cell_id, share_<группа>_<значение>...].
    NaN там, где в клетке нет ни одного объекта группы — доля не определена."""
    env_columns = [c for c in features.columns if c.startswith("env_")]

    shares = pd.DataFrame({"cell_id": features["cell_id"]})
    for group in sorted({_group_of(c) for c in env_columns}):
        columns = [c for c in env_columns if _group_of(c) == group]
        total = features[columns].sum(axis=1)
        for column in columns:
            # деление на 0 даёт NaN — так и надо: доли в пустой группе не существует
            shares[column.replace("env_", "share_")] = features[column] / total.where(total > 0)
    return shares


def wealth_index(features, analytical_cfg: dict):
    """Индекс премиальности в [0,1]: доля премиум-объектов среди премиум+эконом.

    Сглаживание не даёт клетке с одним объектом выдать 0.0 или 1.0.
    Отношение по построению не зависит от плотности — в этом весь смысл."""
    smoothing = analytical_cfg.get("smoothing", 1.0)

    premium = _sum_columns(features, analytical_cfg["premium"])
    econ = _sum_columns(features, analytical_cfg["econ"])
    return (premium + smoothing) / (premium + econ + 2 * smoothing)


def _sum_columns(features, names: list[str]):
    """Сумма счётчиков по списку категорий из config. Отсутствующие колонки — нули."""
    columns = [f"env_{name}" for name in names if f"env_{name}" in features.columns]
    if not columns:
        raise ValueError(f"ни одной колонки из config не нашлось в признаках: {names}")
    return features[columns].sum(axis=1)


def analytical_wealth(features, cfg: dict):
    """wealth_score из данных, без модели. Слот тот же, что у estimate_cells.

    Возвращает [cell_id, wealth_score, wealth_index]:
      wealth_index — сырой индекс в [0,1], для корреляций (Спирмен к монотонным
                     преобразованиям нечувствителен, поэтому считать надо по нему);
      wealth_score — тот же индекс на шкале 0-10, чтобы подставляться вместо VLM.
    Шкалирование ЛИНЕЙНОЕ и фиксированное, не min-max: min-max зависит от района
    и сделал бы скоры разных городов несравнимыми."""
    index = wealth_index(features, cfg["analytical"])
    return pd.DataFrame({
        "cell_id": features["cell_id"],
        "wealth_score": (index * _SCORE_MAX).round().astype(int),
        "wealth_index": index,
    })
