"""Загрузка конфигурации: global.yaml + cities/<city>.yaml.

Правило из п. 2: все параметры в YAML, в коде ничего не зашито.
Каждый запуск сохраняет копию эффективного конфига (см. run.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent / "config"   # каталог с YAML, не этот модуль


class ConfigError(RuntimeError):
    """Конфиг не найден или не проходит проверку."""


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"не найден конфиг: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_global() -> dict:
    """Параметры, общие для всех городов."""
    return _read_yaml(CONFIG_DIR / "global.yaml")


def load_city(city: str) -> dict:
    """Параметры одного города."""
    return _read_yaml(CONFIG_DIR / "cities" / f"{city}.yaml")


def available_cities() -> list[str]:
    """Имена городов, для которых есть конфиг."""
    return sorted(p.stem for p in (CONFIG_DIR / "cities").glob("*.yaml"))


def load(city: str | None = None) -> dict:
    """Эффективный конфиг: глобальный плюс городской под ключом `city`."""
    cfg: dict[str, Any] = load_global()
    validate_global(cfg)
    if city is not None:
        city_cfg = load_city(city)
        validate_city(city_cfg, city)
        cfg["city"] = city_cfg
    return cfg


def validate_global(cfg: dict) -> None:
    """Проверить то, без чего расчёт молча даст неверный результат."""
    for key in ("k", "T_minutes", "walk_speed_m_per_min", "t_max_minutes",
                "snap_max_m", "grid", "vlm", "alpha"):
        if key not in cfg:
            raise ConfigError(f"в global.yaml нет обязательного ключа '{key}'")

    if cfg["walk_speed_m_per_min"] <= 0:
        raise ConfigError("walk_speed_m_per_min должна быть больше нуля")
    if cfg["T_minutes"] > cfg["t_max_minutes"]:
        # иначе порог покрытия окажется за потолком времени и Cov_T всегда = 100
        raise ConfigError("T_minutes больше t_max_minutes — покрытие выродится")
    if cfg["grid"]["size_m"] <= 0:
        raise ConfigError("grid.size_m должен быть больше нуля")
    if cfg["vlm"]["temperature"] != 0:
        # п. 7.1: воспроизводимость; ненулевая температура ломает повтор прогона
        raise ConfigError("vlm.temperature должна быть 0 (воспроизводимость)")


def validate_city(cfg: dict, city: str) -> None:
    """Проверить конфиг города."""
    if cfg.get("name") != city:
        raise ConfigError(f"в cities/{city}.yaml поле name = {cfg.get('name')!r}")
    if cfg.get("role") not in ("dev", "test", "reserve"):
        raise ConfigError(f"role должен быть dev | test | reserve, а не {cfg.get('role')!r}")
    boundary = cfg.get("boundary") or {}
    if not boundary.get("relation_id"):
        raise ConfigError("не задан boundary.relation_id")
