"""Сквозной прогон города: подготовка → скоринг → варианты → метрики → отчёт.

Это `run-city` из п. 15. Правила сбоев по п. 13: вариант, который не удалось
посчитать, получает статус, а не роняет весь город; статусы идут в отчёт.

Города проверки отдельно защищены: без замороженного протокола запуск на них
запрещён (п. 13), это проверяется до начала работы, а не после.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..costs.timer import CostLog
from ..log import RunLogger
from ..report import city as report_mod
from ..run import Run
from ..scoring import client as scoring_client
from ..scoring import prompt as prompt_mod
from .evaluate import run_variants
from .prepare import prepare_city


class ProtocolRequired(RuntimeError):
    """Город проверки без замороженного протокола запускать нельзя."""


def guard_test_city(cfg: dict, protocol_path: Path | None) -> None:
    """Запрет запуска на городах проверки без протокола (п. 13).

    Проверяется ДО работы: узнать об этом после прогона уже поздно —
    результат придётся выбросить."""
    if cfg["city"].get("role") != "test":
        return
    if protocol_path is None or not Path(protocol_path).is_file():
        raise ProtocolRequired(
            f"город {cfg['city']['name']} помечен как test, но протокол не "
            "заморожен: сначала `python -m seto freeze-protocol`")


def _score_variant(mode: str, cell_ids, inputs, cfg: dict, run_dir: Path,
                   costs: CostLog, logger) -> tuple[np.ndarray | None, dict]:
    """Скоры одного варианта. Сбой не роняет город — вариант станет failed."""
    variant = "C" if mode == "text" else "D"
    # кэш должен лежать рядом с raw/ и runs/, а не в текущем каталоге:
    # иначе запуск из другой папки заново тратит тысячи вызовов модели
    from .. import settings as settings_mod
    cache = (settings_mod.data_root(cfg)
             / cfg["paths"].get("vlm_cache", "cache/vlm") / mode)
    try:
        with costs.stage("scoring", f"variant:{variant}") as extra:
            scores, report = scoring_client.score_cells(
                cell_ids, inputs, mode, cfg, cache, run_dir / f"vlm_raw_{mode}")
            scoring_client.check_imputed(report, cfg)
            extra.update(report.as_dict())
        scores.to_parquet(run_dir / f"scores_{mode}.parquet")
        logger.info(f"скоринг {variant}", **report.as_dict())
        return scores["score"].to_numpy(), report.as_dict()
    except Exception as error:
        logger.error(f"скоринг {variant} не удался",
                     reason=f"{type(error).__name__}: {error}")
        return None, {"status": "failed", "reason": str(error)}


def run_city(city: str, cfg: dict, protocol_path: Path | None = None,
             with_image: bool = False, mode: str = "onboard_city") -> Path:
    """Полный прогон города. Возвращает папку прогона."""
    guard_test_city(cfg, protocol_path)
    costs = CostLog(mode=mode)

    with Run.start(city, "run-city", cfg) as run:
        logger = RunLogger(run.run_id, run.dir / "log.jsonl")
        logger.info("старт", city=city, role=cfg["city"].get("role"))

        data = prepare_city(city, cfg, run.dir, logger, costs)
        for path in data.save(run.dir).values():
            run.record_output(path)

        scores: dict[str, np.ndarray] = {}
        scoring_reports: dict[str, dict] = {}

        texts = prompt_mod.describe_all(data.features).tolist()
        values, report = _score_variant("text", data.features["cell_id"].tolist(),
                                        texts, cfg, run.dir, costs, logger)
        scoring_reports["text"] = report
        if values is not None:
            scores["text"] = values

        if with_image:
            images = _load_images(run.dir, data.features["cell_id"].tolist())
            if images is None:
                scoring_reports["image"] = {
                    "status": "missing", "reason": "снимки не загружены (п. 5, Л6)"}
                logger.warn("вариант D пропущен: нет снимков")
            else:
                values, report = _score_variant(
                    "image", data.features["cell_id"].tolist(), images, cfg,
                    run.dir, costs, logger)
                scoring_reports["image"] = report
                if values is not None:
                    scores["image"] = values
        else:
            scoring_reports["image"] = {
                "status": "skipped", "reason": "запуск без --with-image"}

        evaluation = run_variants(data, scores, cfg, costs, logger)
        _save_solutions(evaluation, run.dir)

        costs_path = costs.save(run.dir / "costs.json")
        run.record_output(costs_path)
        paths = report_mod.build(evaluation, data.report, cfg, run.dir,
                                 costs=json.loads(costs_path.read_text("utf-8")),
                                 scoring=scoring_reports)
        for path in paths.values():
            run.record_output(path)
        logger.info("готово", seconds=round(costs.total(), 1))
        logger.close()
    return run.dir


def _save_solutions(evaluation, run_dir: Path) -> None:
    payload = {"variants": [v.as_dict() for v in evaluation.variants]}
    (run_dir / "solutions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    random_result = next((v for v in evaluation.variants if v.name == "A"), None)
    if random_result is not None and random_result.distribution is not None:
        pd.DataFrame({"dCov": random_result.distribution}).to_parquet(
            run_dir / "random_A.parquet")


def _load_images(run_dir: Path, cell_ids) -> list[bytes] | None:
    """Снимки клеток, если они загружены (п. 5). Иначе None."""
    folder = run_dir.parent.parent / "imagery"
    if not folder.is_dir():
        return None
    images = []
    for cell_id in cell_ids:
        path = folder / f"{cell_id}.png"
        if not path.is_file():
            return None
        images.append(path.read_bytes())
    return images
