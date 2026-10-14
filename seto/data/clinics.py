"""Выделение базовой сети: государственные взрослые поликлиники ПМСП.

Пропозал (п. 3.1) требует именно их: «взрослая поликлиника, первичная
медико-санитарная помощь», сеть государственная. По тегам OSM это не
выделяется — замер на 1181 объекте показал, что `operator` заполнен у 6,1 %,
а `operator:type` у двух объектов. Поэтому классификация идёт по названию.

Это ПРОКСИ, а не истина, и код не делает вид, что это истина: каждое
исключение записывается с правилом и примерами, а спорные случаи попадают
в очередь ручной проверки. Пока нет реестра медорганизаций (Л7), ограничение
обязано быть видно в протоколе.

Почему важно отсечь не-взрослые объекты: базовая сеть входит в время до
ближайшей поликлиники, то есть во все метрики и в целевую функцию каждого
варианта. Плюс правило «не ближе 500 м к существующей» вырезает площадки
вокруг каждого объекта — детская поликлиника запрещала бы ставить рядом
взрослую, хотя взрослого она не обслуживает.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Selection:
    """Результат отбора: что взяли, что выбросили и почему."""
    frame: pd.DataFrame
    candidates: int = 0
    excluded_by: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    uncertain: pd.DataFrame | None = None
    unnamed: int = 0

    @property
    def selected(self) -> int:
        return len(self.frame)

    def check(self) -> None:
        """Отобранные плюс исключённые обязаны дать всех кандидатов.

        Если правила начнут пересекаться или считаться не на том множестве,
        расхождение всплывёт здесь, а не в метриках через два этапа."""
        total = self.selected + sum(self.excluded_by.values())
        if total != self.candidates:
            raise ValueError(
                f"отбор не сходится: {self.selected} отобрано + "
                f"{sum(self.excluded_by.values())} исключено != "
                f"{self.candidates} кандидатов")

    def as_dict(self) -> dict:
        return {"candidates": self.candidates, "selected": self.selected,
                "excluded_by": self.excluded_by, "unnamed": self.unnamed,
                "uncertain": 0 if self.uncertain is None else len(self.uncertain),
                "examples": {k: v[:3] for k, v in self.examples.items()}}


def _search(names: pd.Series, pattern: str) -> pd.Series:
    """Поиск по названиям; названия приводятся к нижнему регистру заранее.

    Паттерн компилируется через `re` и передаётся готовым: со строковым
    бэкендом Arrow pandas уводит `.str.contains` в движок RE2, где `\\w` не
    считает кириллицу буквой, и правило молча перестаёт срабатывать. Готовый
    паттерн заставляет pandas использовать движок Python на любом бэкенде."""
    compiled = re.compile(pattern)
    return names.map(lambda value: bool(compiled.search(value))).astype(bool)


def classify(names: pd.Series, rules: dict) -> pd.DataFrame:
    """Разметить названия по правилам.

    Возвращает колонки `is_candidate` и `excluded_by` (имя первого
    сработавшего правила или пустая строка). Порядок правил важен: он задан
    в конфиге и фиксируется в протоколе."""
    lowered = names.fillna("").str.lower()
    candidate = _search(lowered, rules["include"])

    excluded_by = pd.Series("", index=names.index, dtype=object)
    for rule_name, pattern in rules.get("exclude", {}).items():
        hit = candidate & (excluded_by == "") & _search(lowered, pattern)
        excluded_by[hit] = rule_name

    return pd.DataFrame({"is_candidate": candidate, "excluded_by": excluded_by})


def select(medical: pd.DataFrame, rules: dict,
           name_column: str = "name") -> Selection:
    """Отобрать базовую сеть и объяснить каждое исключение.

    Объекты без названия в сеть не попадают: по пустому имени нельзя
    утверждать, что это взрослая государственная поликлиника. Их число
    видно в счётчике `без названия`."""
    names = medical[name_column] if name_column in medical else pd.Series(
        "", index=medical.index)
    marks = classify(names, rules)

    candidates = medical[marks["is_candidate"]].copy()
    reasons = marks.loc[marks["is_candidate"], "excluded_by"]

    excluded_by: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for rule_name in rules.get("exclude", {}):
        hit = reasons == rule_name
        excluded_by[rule_name] = int(hit.sum())
        examples[rule_name] = (candidates.loc[hit, name_column]
                               .dropna().head(5).tolist())

    kept = candidates[reasons == ""].reset_index(drop=True)

    selection = Selection(frame=kept, candidates=int(marks["is_candidate"].sum()),
                          excluded_by=excluded_by, examples=examples,
                          # объекты без названия кандидатами и не были: правилом
                          # их не исключали, поэтому в excluded_by им не место —
                          # иначе числа перестают сходиться
                          unnamed=int((names.fillna("") == "").sum()))
    selection.uncertain = uncertain_cases(kept, rules, name_column)
    return selection


def uncertain_cases(kept: pd.DataFrame, rules: dict,
                    name_column: str = "name") -> pd.DataFrame:
    """Объекты, прошедшие фильтр, но не похожие на типовую поликлинику.

    Типовое название московской взрослой ПМСП — «городская поликлиника №N»
    или «поликлиника №N». Всё остальное прошло только потому, что ни одно
    правило исключения не сработало, а это не то же самое, что уверенность.
    Такие строки идут человеку на проверку, а не тихо остаются в сети."""
    if kept.empty or name_column not in kept:
        return kept.head(0)
    typical = _search(kept[name_column].fillna("").str.lower(),
                      rules.get("typical", r"поликлиник\w*\s*№?\s*\d"))
    return kept[~typical].reset_index(drop=True)
