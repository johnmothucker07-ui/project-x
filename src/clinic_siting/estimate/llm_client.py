"""
ЭТАП 2 · Обёртка вызова модели — ПРОВАЙДЕР-НЕЗАВИСИМАЯ.

Ключевой файл для гибкости: модель и эндпоинт берутся из config
(provider / model / base_url). Если Qwen-триал кончится или забанят —
меняешь base_url в config.yaml, весь остальной код не трогаешь.

Использует OpenAI-совместимый клиент (Qwen через DashScope compatible-mode
работает по тому же протоколу, что GPT-4o).
"""
from __future__ import annotations


def make_client(cfg: dict):
    """Создать OpenAI-совместимый клиент по config (base_url + ключ из .env)."""
    raise NotImplementedError


def call_vlm(client, model: str, prompt: str, image_bytes: bytes | None = None,
             temperature: float = 0) -> str:
    """Вызвать модель. image_bytes=None → только текст (первая версия).
    Когда добавим спутник — просто передаём картинку, сигнатура уже готова."""
    raise NotImplementedError
