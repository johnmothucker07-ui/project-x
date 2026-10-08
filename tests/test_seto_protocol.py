"""Тесты п. 5, 10.5, 12, 13, 14: снимки, внешнее решение, протокол, затраты."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.costs import human  # noqa: E402
from seto.experiment import protocol as proto  # noqa: E402
from seto.imagery import tiles  # noqa: E402
from seto.variants import external  # noqa: E402


def _allowed(size: int = 4) -> np.ndarray:
    allowed = np.ones((size, size), dtype=bool)
    np.fill_diagonal(allowed, False)
    return allowed


# --- снимки (п. 5) ---

def test_zoom_grows_with_latitude():
    """Масштаб Web Mercator зависит от широты: один зум даёт разное (п. 5.2)."""
    equator = tiles.zoom_for(0.0, 1.0)
    moscow = tiles.zoom_for(55.75, 1.0)
    assert moscow < equator


def test_resolution_matches_target():
    zoom = tiles.zoom_for(55.75, 1.0)
    assert tiles.resolution_at(55.75, zoom) <= 1.0
    assert tiles.resolution_at(55.75, zoom - 1) > 1.0


def test_tiles_cover_the_bbox():
    bbox = (37.60, 55.74, 37.62, 55.76)
    covering = tiles.tiles_for_bbox(bbox, 16)
    assert tiles.tile_xy(37.60, 55.76, 16) in covering
    assert tiles.tile_xy(37.62, 55.74, 16) in covering


def test_provider_must_be_configured():
    """Подставлять произвольный тайл-сервер нельзя — это Л6."""
    with pytest.raises(tiles.ProviderNotConfigured, match="Л6"):
        tiles.require_provider({"imagery": {"provider": None}})
    assert tiles.require_provider({"imagery": {"provider": "esri"}}) == "esri"


def test_flat_image_is_flagged():
    """Однотонный тайл — признак заглушки провайдера (п. 5.4)."""
    flat = np.zeros((16, 16, 3), dtype=np.uint8)
    assert not tiles.quality_flags(flat)["ok"]
    varied = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    assert tiles.quality_flags(varied)["ok"]


# --- внешнее решение E (п. 10.5) ---

def _solution_file(tmp_path: Path, **overrides) -> Path:
    payload = {"city": "moscow", "data_manifest_sha256": "abc", "k": 2,
               "sites": [0, 2], "method_version": "ahp-1"}
    payload.update(overrides)
    path = tmp_path / "solution_E.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accepts_valid_solution(tmp_path):
    result = external.import_solution(_solution_file(tmp_path), _allowed(),
                                      {"k": 2}, "moscow", "abc")
    assert result["sites"] == (0, 2)
    assert result["method_version"] == "ahp-1"


def test_rejects_other_city(tmp_path):
    with pytest.raises(external.ExternalSolutionError, match="города"):
        external.import_solution(_solution_file(tmp_path, city="kazan"),
                                 _allowed(), {"k": 2}, "moscow", "abc")


def test_rejects_other_data(tmp_path):
    """Решение, посчитанное на другом снимке, сравнивать нельзя."""
    with pytest.raises(external.ExternalSolutionError, match="других данных"):
        external.import_solution(_solution_file(tmp_path), _allowed(),
                                 {"k": 2}, "moscow", "другой-хеш")


def test_rejects_constraint_violation(tmp_path):
    close = _allowed()
    close[0, 2] = close[2, 0] = False        # пара слишком близко
    with pytest.raises(external.ExternalSolutionError, match="ограничения"):
        external.import_solution(_solution_file(tmp_path), close, {"k": 2},
                                 "moscow", "abc")


def test_rejects_wrong_k(tmp_path):
    with pytest.raises(external.ExternalSolutionError, match="ограничения"):
        external.import_solution(_solution_file(tmp_path, sites=[0]), _allowed(),
                                 {"k": 2}, "moscow", "abc")


def test_invalid_sensitivity_solutions_are_dropped_not_fixed(tmp_path):
    close = _allowed()
    close[1, 3] = close[3, 1] = False
    path = _solution_file(tmp_path,
                          weight_sensitivity_solutions=[[0, 1], [1, 3]])
    result = external.import_solution(path, close, {"k": 2}, "moscow", "abc")
    assert result["sensitivity_solutions"] == [(0, 1)]


# --- протокол (п. 13) ---

def test_verify_detects_changed_parameter(tmp_path, monkeypatch):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})
    cfg = settings.load()
    json_path, _ = proto.freeze(cfg, {"moscow": "dev"}, {}, tmp_path)

    assert proto.verify(json_path, cfg)["ok"]
    changed = settings.load()
    changed["k"] = 5
    result = proto.verify(json_path, changed)
    assert not result["ok"]
    assert any("параметр k" in p for p in result["problems"])


def test_verify_detects_other_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})
    cfg = settings.load()
    json_path, _ = proto.freeze(cfg, {"moscow": "dev"}, {}, tmp_path)

    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c2", "branch": "main", "dirty": False})
    assert not proto.verify(json_path, cfg)["ok"]


def test_cannot_freeze_on_dirty_tree(tmp_path, monkeypatch):
    """Иначе протокол сослался бы на коммит, которого нет в истории."""
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": True})
    with pytest.raises(proto.ProtocolMismatch, match="грязное"):
        proto.freeze(settings.load(), {"moscow": "dev"}, {}, tmp_path)


def test_protocol_records_what_is_not_computed(tmp_path, monkeypatch):
    monkeypatch.setattr(proto, "git_state",
                        lambda: {"commit": "c1", "branch": "main", "dirty": False})
    json_path, md_path = proto.freeze(settings.load(), {"moscow": "dev"}, {},
                                      tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert any("δ" in item for item in payload["not_computed"])
    assert "δ" in md_path.read_text(encoding="utf-8")


# --- затраты (п. 14) ---

def test_human_log_accumulates(tmp_path):
    path = tmp_path / "human_log.csv"
    human.append(human.HumanEntry("moscow", "D", "onboard_city", 30, "проверка"),
                 path)
    human.append(human.HumanEntry("moscow", "D", "onboard_city", 15, "правки"),
                 path)
    table = human.summary(path)
    assert float(table.loc[table.variant == "D", "minutes"].iloc[0]) == 45.0


def test_cost_criterion_without_reference_is_no_data(tmp_path):
    """Отсутствие данных по E — «нет данных», а не «выполнен»."""
    path = tmp_path / "human_log.csv"
    human.append(human.HumanEntry("moscow", "D", "onboard_city", 30, "x"), path)
    assert human.cost_criterion(path, "moscow")["status"] == "нет данных"


def test_cost_criterion_compares_to_half_of_reference(tmp_path):
    path = tmp_path / "human_log.csv"
    human.append(human.HumanEntry("moscow", "E", "onboard_city", 100, "x"), path)
    human.append(human.HumanEntry("moscow", "D", "onboard_city", 40, "y"), path)
    assert human.cost_criterion(path, "moscow")["status"] == "выполнен"


def test_price_of_gain_not_computed_without_gain():
    assert human.price_of_gain(100, 50, 0)["status"] == "не считается"
    result = human.price_of_gain(100, 50, 1000)
    assert result["minutes_per_1000_people"] == pytest.approx(50.0)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        human.HumanEntry("moscow", "D", "что-то", 10, "x").as_row()
