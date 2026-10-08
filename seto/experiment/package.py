"""Пакет результатов (п. 18.5).

Требование пропозала: «РЕЗУЛЬТАТ = код сравнения, сохранённые входы и решения,
из которых метрики и выводы пересчитываются». Значит пакет должен позволять
пересчёт БЕЗ интернета — по сохранённому кэшу ответов модели.

Сырые входы (PBF, растр, снимки) в пакет не кладутся: это гигабайты. Вместо
них идут ссылки и хеши, а сами файлы уходят в архив лаборатории (Л17).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..run import git_state, sha256_of

# что копируется в пакет: всё, из чего пересчитываются выводы
ARTIFACTS = ("metrics.parquet", "metrics_main.csv", "metrics_thresholds.csv",
             "differences.csv", "solutions.json", "random_A.parquet",
             "report.json", "report.md", "costs.json", "manifest.json",
             "city_report.json", "effective_config.yaml", "log.jsonl",
             "scores_text.parquet", "scores_image.parquet", "solution_E.json")

# сырые входы: только ссылки и хеши, сами файлы в архив лаборатории
BY_REFERENCE = ("osm.pbf", ".tif", "imagery")


def build(run_dirs: list[Path], out_dir: Path, protocol: Path | None = None,
          raw_dirs: list[Path] | None = None) -> dict:
    """Собрать пакет из прогонов городов."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "cities": {},
        "raw_inputs": [],
        "protocol": None,
    }

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        city = run_dir.parent.name
        target = out_dir / "runs" / city / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in ARTIFACTS:
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)
                copied.append(name)
        vlm_raw = [d for d in run_dir.glob("vlm_raw_*") if d.is_dir()]
        for folder in vlm_raw:
            shutil.copytree(folder, target / folder.name, dirs_exist_ok=True)
            copied.append(folder.name + "/")
        manifest["cities"][city] = {"run_id": run_dir.name, "files": copied}

    for raw in (raw_dirs or []):
        raw = Path(raw)
        for path in sorted(raw.rglob("*")):
            if path.is_file() and any(mark in path.name or mark in str(path)
                                      for mark in BY_REFERENCE):
                manifest["raw_inputs"].append({
                    "path": str(path), "size_bytes": path.stat().st_size,
                    "sha256": sha256_of(path)})

    if protocol is not None and Path(protocol).is_file():
        shutil.copy2(protocol, out_dir / "protocol.json")
        manifest["protocol"] = sha256_of(protocol)

    (out_dir / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPRODUCE.md").write_text(_reproduce_text(manifest),
                                          encoding="utf-8")
    return manifest


def _reproduce_text(manifest: dict) -> str:
    cities = ", ".join(manifest["cities"]) or "—"
    return f"""# Как пересчитать результаты

Пакет собран {manifest['built_at']} на коммите `{manifest['git'].get('commit')}`.
Города: {cities}.

## Что внутри

- `runs/<город>/<run_id>/` — метрики, решения, отчёты, журналы, сырые ответы
  модели (`vlm_raw_*`);
- `protocol.json` — замороженный протокол, если он был;
- `package_manifest.json` — хеши сырых входов, которые в пакет не вошли.

## Пересчёт без интернета

Сырые ответы модели лежат в пакете, а кэш скоринга ключуется по входу, так что
повторный прогон не обращается к API:

```bash
git checkout {manifest['git'].get('commit')}
python -m seto run-city <город> --mode recompute
```

Метрики должны совпасть с `metrics.parquet` побайтно. Если не совпали — это
находка: что-то зависит от окружения, и это надо зафиксировать.

## Чего в пакете НЕТ

Сырые входы — выгрузка OSM, растр населения, спутниковые снимки — весят
гигабайты и лежат в архиве лаборатории. Их хеши в `package_manifest.json`,
по ним проверяется, что архив тот самый.
"""
