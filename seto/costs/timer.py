"""Учёт времени и расходов по этапам (п. 14).

Пропозал меряет не только качество, но и цену: половина вывода — про затраты.
Поэтому время считается не «на глазок в конце», а таймером на каждом этапе,
с тегом общей подготовки или конкретного варианта.

Теги важны: подготовка данных (сетка, население, граф, площадки) считается
ОДИН раз и показывается отдельно, а варианту приписываются только его
дополнительные затраты — вызовы модели для C и D, время решателя для B и M.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

SHARED = "shared_prep"


@dataclass
class Entry:
    """Один замер."""
    stage: str
    tag: str              # shared_prep или variant:<X>
    mode: str             # dev_build | onboard_city | recompute
    seconds: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"stage": self.stage, "tag": self.tag, "mode": self.mode,
                "seconds": round(self.seconds, 2), **self.extra}


class CostLog:
    """Журнал затрат прогона."""

    def __init__(self, mode: str = "onboard_city") -> None:
        self.mode = mode
        self.entries: list[Entry] = []

    @contextmanager
    def stage(self, name: str, tag: str = SHARED, **extra):
        """Засечь этап. Время пишется и при падении — иначе непонятно,
        на чём прогон встал и сколько успел потратить."""
        started = time.perf_counter()
        payload: dict = dict(extra)
        try:
            yield payload
        finally:
            self.entries.append(Entry(name, tag, self.mode,
                                      time.perf_counter() - started, payload))

    def total(self, tag: str | None = None) -> float:
        return sum(e.seconds for e in self.entries if tag is None or e.tag == tag)

    def as_table(self) -> list[dict]:
        return [e.as_dict() for e in self.entries]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": self.mode,
                   "total_seconds": round(self.total(), 2),
                   "by_tag": {tag: round(self.total(tag), 2)
                              for tag in sorted({e.tag for e in self.entries})},
                   "stages": self.as_table()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path
