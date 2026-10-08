"""Журнал человеко-времени и ошибок (п. 14).

Половина вывода пропозала — про затраты, и ключевой критерий считается
по АКТИВНОМУ ВРЕМЕНИ ЧЕЛОВЕКА, а не по времени машины. Машинное время
меряется таймером, человеческое записать может только человек.

Отдельно важно, что API лаборатории бесплатный (Д3): стоимость вызовов
равна нулю, и весь вес критерия затрат ложится на человеко-время. Поэтому
журнал — не формальность, а половина результата.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ("timestamp", "city", "variant", "mode", "minutes", "activity", "note")
MODES = ("dev_build", "onboard_city", "recompute")


@dataclass
class HumanEntry:
    city: str
    variant: str
    mode: str
    minutes: float
    activity: str
    note: str = ""

    def as_row(self) -> dict:
        if self.mode not in MODES:
            raise ValueError(f"режим должен быть одним из {MODES}")
        if self.minutes <= 0:
            raise ValueError("время должно быть больше нуля")
        return {"timestamp": datetime.now(timezone.utc).isoformat(),
                "city": self.city, "variant": self.variant, "mode": self.mode,
                "minutes": self.minutes, "activity": self.activity,
                "note": self.note}


def append(entry: HumanEntry, path: Path) -> Path:
    """Дописать запись. Файл общий и накапливается весь эксперимент."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(entry.as_row())
    return path


def summary(path: Path) -> pd.DataFrame:
    """Сводка по городу и варианту: сколько активного времени потрачено."""
    if not Path(path).is_file():
        return pd.DataFrame(columns=["city", "variant", "mode", "minutes"])
    frame = pd.read_csv(path)
    return (frame.groupby(["city", "variant", "mode"], as_index=False)["minutes"]
            .sum().sort_values(["city", "variant", "mode"]))


def cost_criterion(path: Path, city: str, reference: str = "E",
                   candidate: str = "D", ratio: float = 0.5) -> dict:
    """Критерий затрат (п. 18.3): время человека у D не больше половины от E.

    Если данных по E нет — «нет данных», а не «выполнен»: отсутствие
    сравнения не является его прохождением."""
    table = summary(path)
    subset = table[(table["city"] == city) & (table["mode"] == "onboard_city")]
    times = subset.set_index("variant")["minutes"].to_dict()

    if reference not in times:
        return {"status": "нет данных",
                "reason": f"нет записей времени для варианта {reference} (Л18)",
                "candidate_minutes": times.get(candidate)}
    if candidate not in times:
        return {"status": "нет данных",
                "reason": f"нет записей времени для варианта {candidate}"}

    limit = ratio * times[reference]
    return {"status": "выполнен" if times[candidate] <= limit else "не выполнен",
            "candidate_minutes": times[candidate],
            "reference_minutes": times[reference], "limit_minutes": limit}


def price_of_gain(costs_minutes: float, reference_minutes: float,
                  delta_people: float) -> dict:
    """Цена прироста: сколько минут человека на одного жителя (п. 14).

    Считается только при положительном приросте: делить на ноль или на
    отрицательное число бессмысленно, и такой «показатель» вводил бы в
    заблуждение."""
    if delta_people <= 0:
        return {"status": "не считается",
                "reason": "прирост не положительный"}
    extra = costs_minutes - reference_minutes
    return {"status": "посчитана",
            "extra_minutes": extra,
            "delta_people": delta_people,
            "minutes_per_1000_people": 1000 * extra / delta_people}
