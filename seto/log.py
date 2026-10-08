"""Структурированный лог: по JSON-строке на событие, в каждой — run_id.

Нужен, чтобы разбирать сбои задним числом (п. 13: сбой варианта — один перезапуск,
потом статус failed). По обычному текстовому логу этого не восстановить.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class RunLogger:
    """Пишет события в runs/<city>/<run_id>/log.jsonl и в консоль."""

    def __init__(self, run_id: str, path: Path | None = None,
                 echo: TextIO | None = sys.stderr) -> None:
        self.run_id = run_id
        self._echo = echo
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def event(self, level: str, message: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "level": level,
            "message": message,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()   # обрыв не должен терять последние строки
        if self._echo is not None:
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[{level}] {message}" + (f"  {extra}" if extra else ""),
                  file=self._echo, flush=True)

    def info(self, message: str, **fields: Any) -> None:
        self.event("info", message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self.event("warn", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.event("error", message, **fields)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
