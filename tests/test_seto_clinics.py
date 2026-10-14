"""Тесты отбора базовой сети: государственные взрослые поликлиники ПМСП."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data import clinics as CL  # noqa: E402

RULES = {
    "include": "поликлин",
    "typical": r"поликлиник\w*\s*№?\s*\d",
    "exclude": {
        "детская": "детск|педиатр",
        "ведомственная": "миноборон|министерств|войсковой|мид\\b",
        "частная": "ооо|\\.ру\\b",
    },
}


def frame(*names) -> pd.DataFrame:
    return pd.DataFrame({"name": list(names)})


# --- что попадает в сеть ---

def test_typical_city_polyclinic_is_selected():
    sel = CL.select(frame("Городская поликлиника №191"), RULES)
    assert sel.selected == 1


def test_branch_is_selected_too():
    """Филиал — отдельная точка обслуживания, он обслуживает жителей."""
    sel = CL.select(frame("Поликлиника №115 филиал №2"), RULES)
    assert sel.selected == 1


def test_non_clinic_is_not_a_candidate():
    sel = CL.select(frame("Городская больница №1", "Аптека"), RULES)
    assert sel.candidates == 0 and sel.selected == 0


# --- что отсекается ---

def test_children_clinic_is_excluded():
    """Детская поликлиника взрослого не обслуживает, но при этом запрещала бы
    ставить взрослую рядом по правилу 500 м."""
    sel = CL.select(frame("Детская городская поликлиника №30"), RULES)
    assert sel.selected == 0
    assert sel.excluded_by["детская"] == 1


def test_departmental_clinic_is_excluded():
    sel = CL.select(frame("Центральная поликлиника №5 Министерства обороны РФ"),
                    RULES)
    assert sel.excluded_by["ведомственная"] == 1


def test_private_clinic_is_excluded():
    sel = CL.select(frame("Поликлиника.ру"), RULES)
    assert sel.excluded_by["частная"] == 1


def test_first_matching_rule_wins():
    """Порядок правил задан конфигом; объект считается ровно один раз."""
    sel = CL.select(frame("Детская поликлиника ООО Здоровье"), RULES)
    assert sel.excluded_by["детская"] == 1
    assert sel.excluded_by["частная"] == 0


# --- учёт ---

def test_counts_add_up():
    sel = CL.select(frame("Поликлиника №1", "Детская поликлиника №2",
                          "Поликлиника.ру", "Аптека"), RULES)
    sel.check()                     # не бросает
    assert sel.candidates == 3 and sel.selected == 1


def test_check_catches_broken_accounting():
    sel = CL.select(frame("Поликлиника №1"), RULES)
    sel.excluded_by["выдумка"] = 5
    with pytest.raises(ValueError, match="не сходится"):
        sel.check()


def test_unnamed_objects_are_counted_apart_from_rules():
    """Безымянные кандидатами не были — в excluded_by им не место,
    иначе числа перестают сходиться."""
    sel = CL.select(frame("Поликлиника №1", None, None), RULES)
    sel.check()
    assert sel.unnamed == 2
    assert "без названия" not in sel.excluded_by


# --- очередь ручной проверки ---

def test_unusual_name_goes_to_manual_review():
    """Ведомственные ловятся не все: «Поликлиника ФТС РФ» правила проходит.
    Такие строки обязаны попасть человеку, а не осесть в сети молча."""
    sel = CL.select(frame("Поликлиника №17", "Поликлиника ФТС РФ"), RULES)
    assert sel.selected == 2
    assert sel.uncertain["name"].tolist() == ["Поликлиника ФТС РФ"]


def test_typical_name_is_not_flagged():
    sel = CL.select(frame("Городская поликлиника № 220 филиал № 3"), RULES)
    assert len(sel.uncertain) == 0


# --- регрессия: движок регулярок ---

def test_cyrillic_word_boundary_works():
    """pandas со строковым бэкендом Arrow уводит .str.contains в RE2, где
    \\w не считает кириллицу буквой, и правило молча перестаёт срабатывать."""
    found = CL._search(pd.Series(["поликлиника №17"]), r"поликлиник\w*\s*№?\s*\d")
    assert bool(found.iloc[0])


def test_case_is_ignored():
    sel = CL.select(frame("ПОЛИКЛИНИКА №17", "детская Поликлиника №3"), RULES)
    assert sel.selected == 1 and sel.excluded_by["детская"] == 1
