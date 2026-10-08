"""Тесты каркаса (Ш1): конфиги, сетка в UTM, манифест прогона."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.geo import cell_id, cell_index, grid_origin, utm_epsg  # noqa: E402
from seto.run import Run, sha256_of  # noqa: E402


# --- конфиги ---

def test_global_loads_and_validates():
    cfg = settings.load()
    assert cfg["grid"]["size_m"] == 500          # размер как в MVP (пропозал п. 3.1)
    assert cfg["vlm"]["temperature"] == 0        # воспроизводимость (п. 7.1)
    assert cfg["min_dist_between_new_m"] == 500  # разнесение 1,2 км убрано в v4


def test_city_loads():
    cfg = settings.load("moscow")
    assert cfg["city"]["role"] in ("dev", "test", "reserve")
    assert cfg["city"]["boundary"]["relation_id"]


def test_moscow_is_listed():
    assert "moscow" in settings.available_cities()


def test_missing_city_raises():
    with pytest.raises(settings.ConfigError):
        settings.load("atlantis")


def test_key_is_not_in_config():
    """Ключ модели в конфиге храниться не должен — только имя переменной."""
    text = (Path(settings.CONFIG_DIR) / "global.yaml").read_text(encoding="utf-8")
    assert "sk-" not in text


def test_threshold_above_tmax_rejected():
    cfg = settings.load_global()
    cfg["T_minutes"] = cfg["t_max_minutes"] + 1
    with pytest.raises(settings.ConfigError):
        settings.validate_global(cfg)


# --- геометрия ---

def test_utm_zone_for_moscow():
    assert utm_epsg(37.6, 55.75) == 32637      # зона 37N


def test_utm_southern_hemisphere():
    assert utm_epsg(-58.4, -34.6) == 32721     # Буэнос-Айрес, зона 21S


def test_origin_rounds_down_to_cell_size():
    assert grid_origin(412_345.0, 6_181_777.0, 500) == (412_000.0, 6_181_500.0)


def test_origin_stable_when_boundary_shifts():
    """Смысл решения Д5: сдвиг границы не должен двигать сетку."""
    assert grid_origin(412_345.0, 6_181_777.0, 500) == \
           grid_origin(412_355.0, 6_181_787.0, 500)


def test_cell_index_and_id():
    origin = (412_000.0, 6_181_500.0)
    assert cell_index(412_000.0, 6_181_500.0, origin, 500) == (0, 0)
    assert cell_index(412_700.0, 6_182_100.0, origin, 500) == (1, 1)
    assert cell_id(1, 1, columns=10, id_format="cell_{:06d}") == "cell_000011"


def test_cell_id_rejects_column_outside_grid():
    with pytest.raises(ValueError):
        cell_id(10, 0, columns=10, id_format="cell_{:06d}")


# --- прогон ---

def test_run_writes_manifest(tmp_path):
    cfg = settings.load("moscow")
    with Run.start("moscow", "test", cfg, root=tmp_path) as run:
        payload = run.dir / "out.txt"
        payload.write_text("данные", encoding="utf-8")
        run.record_output(payload)

    manifest = json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["city"] == "moscow"
    assert manifest["command"] == "test"
    assert "git" in manifest and "packages" in manifest
    assert str(payload) in manifest["outputs"]
    assert (run.dir / "effective_config.yaml").is_file()


def test_run_never_overwrites(tmp_path):
    """Папка прогона создаётся с exist_ok=False — повтор должен падать."""
    cfg = settings.load("moscow")
    run = Run.start("moscow", "test", cfg, root=tmp_path)
    with pytest.raises(FileExistsError):
        run.dir.mkdir(parents=True, exist_ok=False)


def test_manifest_written_even_on_failure(tmp_path):
    """Манифест нужен и при падении: иначе непонятно, на чём оборвались."""
    cfg = settings.load("moscow")
    try:
        with Run.start("moscow", "test", cfg, root=tmp_path) as run:
            raise RuntimeError("сбой посреди прогона")
    except RuntimeError:
        pass
    assert (run.dir / "manifest.json").is_file()


def test_sha256_matches_known_value(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    assert sha256_of(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
