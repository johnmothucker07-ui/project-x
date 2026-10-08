"""Тесты п. 7 и 9: промпты, разбор ответа, кэш, веса спроса."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.scoring import client as CL, prompt as P, weights as W  # noqa: E402


# --- промпты ---

def test_prompts_differ_only_in_input_clause():
    """Иначе разница D − C мерила бы формулировку, а не изображение (п. 7.1)."""
    image = P.build("image", 500)
    text = P.build("text", 500, "описание")
    common = P.TEMPLATE.split("{input_clause}")[1].format(size="500x500")
    assert common in image and common in text
    assert P.IMAGE_CLAUSE in image and P.IMAGE_CLAUSE not in text
    assert "описание" in text


def test_prompt_asks_one_question():
    """Промпт спрашивает одно — нужна ли поликлиника (п. 7.1, решение Д12)."""
    text = P.build("text", 500, "x").lower()
    assert "поликлиника" in text
    assert "wealth" not in text and "премиальн" not in text


def test_prompt_forbids_counting():
    assert "не вычисляй" in P.build("image", 500).lower()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        P.build("audio", 500)


def test_text_mode_requires_text():
    with pytest.raises(ValueError):
        P.build("text", 500)


def test_prompt_hash_is_stable():
    assert P.prompt_hash(P.build("image", 500)) == P.prompt_hash(P.build("image", 500))
    assert P.prompt_hash(P.build("image", 500)) != P.prompt_hash(P.build("text", 500, "a"))


# --- описание клетки ---

def _row(**overrides):
    base = {"built_share": 0.3, "residential_share": 0.2, "public_share": 0.1,
            "buildings_count": 42, "mean_levels": 9.0, "levels_known_share": 0.8,
            "schools": 2, "kindergartens": 3, "pharmacies": 4, "shops": 5,
            "transport_stops": 6, "green_share": 0.05, "water_share": 0.0,
            "landuse_industrial_share": 0.0, "street_density_m_per_km2": 12000}
    base.update(overrides)
    return pd.Series(base)


def test_description_has_no_population():
    """Население уже в весе w_i, и модель не должна считать числа."""
    text = P.describe(_row()).lower()
    assert "населен" not in text and "жител" not in text


def test_missing_values_say_so_explicitly():
    """Пропуск — словом, не нулём: ноль школ и неизвестно сколько — разное."""
    text = P.describe(_row(mean_levels=np.nan, schools=None))
    assert text.count("нет данных") >= 2
    assert "Школы: нет данных" in text


def test_description_is_deterministic():
    assert P.describe(_row()) == P.describe(_row())


# --- разбор ответа ---

def test_parses_plain_json():
    assert CL.parse_response('{"score": 7, "rationale": "плотная застройка"}') == {
        "score": 7, "rationale": "плотная застройка"}


def test_parses_markdown_wrapped():
    raw = '```json\n{"score": 3, "rationale": "окраина"}\n```'
    assert CL.parse_response(raw)["score"] == 3


def test_empty_content_is_an_error_not_zero():
    """У reasoning-моделей content бывает пустым, когда рассуждение съело лимит."""
    for value in (None, "", "   "):
        with pytest.raises(CL.ScoreSchemaError, match="пустой"):
            CL.parse_response(value)


def test_score_outside_scale_is_rejected():
    with pytest.raises(CL.ScoreSchemaError, match="вне шкалы"):
        CL.parse_response('{"score": 11}')
    with pytest.raises(CL.ScoreSchemaError, match="вне шкалы"):
        CL.parse_response('{"score": -1}')


def test_non_integer_score_is_rejected():
    with pytest.raises(CL.ScoreSchemaError, match="не целое"):
        CL.parse_response('{"score": "высокий"}')


def test_missing_score_is_rejected():
    with pytest.raises(CL.ScoreSchemaError, match="нет поля score"):
        CL.parse_response('{"rationale": "что-то"}')


def test_broken_json_is_rejected():
    with pytest.raises(CL.ScoreSchemaError):
        CL.parse_response("вообще не json")


# --- кэш ---

def test_cache_key_depends_on_input():
    """Мина MVP: ключ не учитывал вход, и снимок подхватывал текстовый ответ."""
    prompt = P.build("image", 500)
    first = CL.cache_key("cell_000001", "m", prompt, "image", b"\x01\x02")
    second = CL.cache_key("cell_000001", "m", prompt, "image", b"\x03\x04")
    assert first != second


def test_cache_key_depends_on_cell_model_and_prompt():
    prompt = P.build("text", 500, "a")
    other = P.build("text", 500, "b")
    base = CL.cache_key("cell_1", "m1", prompt, "text", "a")
    assert base != CL.cache_key("cell_2", "m1", prompt, "text", "a")
    assert base != CL.cache_key("cell_1", "m2", prompt, "text", "a")
    assert base != CL.cache_key("cell_1", "m1", other, "text", "a")


def test_cached_cells_do_not_call_the_api(tmp_path):
    """Повторный запуск не ходит в API — он же даёт возобновление после обрыва."""
    cfg = {"vlm": {"model": "m", "temperature": 0, "max_tokens": 10,
                   "max_parallel": 1, "retries": 1,
                   "base_url_env": "NO_URL", "api_key_env": "NO_KEY"},
           "grid": {"size_m": 500}}
    cell_ids, texts = ["cell_000001"], ["описание"]
    prompt = P.build("text", 500, texts[0])
    key = CL.cache_key(cell_ids[0], "m", prompt, "text", texts[0])
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"score": 5, "rationale": "из кэша"}), encoding="utf-8")

    scores, report = CL.score_cells(cell_ids, texts, "text", cfg, tmp_path)
    assert scores["score"].iloc[0] == 5
    assert report.from_cache == 1 and report.api_calls == 0


class _BrokenClient:
    """Клиент, у которого любой вызов падает — так проверяется путь imputed."""

    class chat:                                     # noqa: N801
        class completions:                          # noqa: N801
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("модель недоступна")


def test_failed_cells_get_median_and_are_flagged(tmp_path, monkeypatch):
    """Сбойная клетка получает медиану города и флаг imputed (п. 7.1)."""
    monkeypatch.setattr(CL, "_make_client", lambda _cfg: _BrokenClient())
    cfg = {"vlm": {"model": "m", "temperature": 0, "max_tokens": 10,
                   "max_parallel": 1, "retries": 0,
                   "base_url_env": "NO_URL", "api_key_env": "NO_KEY"},
           "grid": {"size_m": 500}}
    ids = ["a", "b", "c"]
    for cell_id, score in zip(ids[:2], (4, 8)):
        prompt = P.build("text", 500, cell_id)
        key = CL.cache_key(cell_id, "m", prompt, "text", cell_id)
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"score": score, "rationale": ""}), encoding="utf-8")

    scores, report = CL.score_cells(ids, ids, "text", cfg, tmp_path)
    assert report.imputed == 1                      # третья клетка не в кэше
    assert scores.loc[scores.cell_id == "c", "score"].iloc[0] == 6.0
    assert bool(scores.loc[scores.cell_id == "c", "imputed"].iloc[0])
    assert report.failures and "модель недоступна" in report.failures[0]


def test_missing_credentials_fail_loudly(tmp_path, monkeypatch):
    """Отсутствие ключа — ошибка настройки, а не повод пометить всё imputed."""
    monkeypatch.delenv("NO_URL", raising=False)
    monkeypatch.delenv("NO_KEY", raising=False)
    cfg = {"vlm": {"model": "m", "temperature": 0, "max_tokens": 10,
                   "max_parallel": 1, "retries": 0,
                   "base_url_env": "NO_URL", "api_key_env": "NO_KEY"},
           "grid": {"size_m": 500}}
    with pytest.raises(RuntimeError, match="не заданы"):
        CL.score_cells(["a"], ["текст"], "text", cfg, tmp_path)


def test_too_many_imputed_fails_the_variant():
    report = CL.ScoringReport(cells=100, imputed=6)
    with pytest.raises(RuntimeError, match="неуспешен"):
        CL.check_imputed(report, {"imputation_max_share": 0.05})
    CL.check_imputed(CL.ScoringReport(cells=100, imputed=5),
                     {"imputation_max_share": 0.05})


# --- веса спроса ---

def test_normalize_maps_to_unit_range():
    normalized, degenerate = W.normalize(np.array([2.0, 6.0, 10.0]))
    assert normalized.tolist() == [0.0, 0.5, 1.0]
    assert not degenerate


def test_identical_scores_are_flagged_and_give_zeros():
    """Модель не различила клетки — вариант сводится к B, и это помечается."""
    normalized, degenerate = W.normalize(np.array([7.0, 7.0, 7.0]))
    assert normalized.tolist() == [0.0, 0.0, 0.0]
    assert degenerate


def test_variant_b_weight_is_population():
    population = np.array([100.0, 200.0])
    assert W.demand(population).tolist() == [100.0, 200.0]


def test_score_multiplies_population():
    population = np.array([100.0, 100.0])
    weights = W.demand(population, np.array([0.0, 10.0]), alpha=1.0)
    assert weights.tolist() == [100.0, 200.0]       # ŝ = 0 и 1


def test_alpha_scales_the_effect():
    population = np.array([100.0, 100.0])
    half = W.demand(population, np.array([0.0, 10.0]), alpha=0.5)
    double = W.demand(population, np.array([0.0, 10.0]), alpha=2.0)
    assert half.tolist() == [100.0, 150.0]
    assert double.tolist() == [100.0, 300.0]


def test_empty_cell_has_zero_weight_whatever_the_score():
    """Скор считается для всех клеток (нужен для нормализации), но спроса нет."""
    weights = W.demand(np.array([0.0, 100.0]), np.array([10.0, 0.0]), alpha=1.0)
    assert weights[0] == 0.0


def test_negative_alpha_is_rejected():
    with pytest.raises(ValueError):
        W.demand(np.array([1.0]), np.array([1.0]), alpha=-1.0)


def test_compare_detects_identical_weights():
    population = np.array([10.0, 20.0, 30.0])
    result = W.compare(population, population)
    assert result["max_relative_diff"] == 0.0
    assert result["correlation"] == pytest.approx(1.0)


def test_compare_reports_relative_difference():
    result = W.compare(np.array([100.0, 100.0]), np.array([100.0, 150.0]))
    assert result["max_relative_diff"] == pytest.approx(0.5)
