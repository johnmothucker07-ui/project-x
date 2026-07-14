"""
ЭТАП 2 · Обёртка вызова модели — ПРОВАЙДЕР-НЕЗАВИСИМАЯ.

Ключевой файл для гибкости: модель и эндпоинт берутся из config
(provider / model / base_url). Если Qwen-триал кончится или забанят —
меняешь base_url в config.yaml, весь остальной код не трогаешь.

Использует OpenAI-совместимый клиент (Qwen через DashScope compatible-mode
работает по тому же протоколу, что GPT-4o).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def make_client(cfg: dict):
    """Создать OpenAI-совместимый клиент по config (base_url + ключ из .env)."""
    load_dotenv()  # подхватить QWEN_API_KEY из .env
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("QWEN_API_KEY не задан — скопируй .env.example в .env и впиши ключ")
    return OpenAI(api_key=api_key, base_url=cfg["estimate"]["base_url"])


def call_vlm(client, model: str, prompt: str, image_bytes: bytes | None = None,
             temperature: float = 0) -> str:
    """Вызвать модель. image_bytes=None → только текст (первая версия).
    Когда добавим спутник — просто передаём картинку, сигнатура уже готова."""
    if image_bytes is not None:
        # передача картинки — это отложенная задача (спутник, MVP-2), см. TODO «ОТЛОЖЕНО»
        raise NotImplementedError("Передача изображения пока не поддержана (задача про спутник)")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content
