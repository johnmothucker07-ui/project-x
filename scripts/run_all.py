"""Прогнать весь пайплайн: grid → data → estimate → optimize → validate.

Запускает скрипты по порядку; при падении любого — останавливается.
Благодаря кэшу повторный прогон Estimate не тратит токены.
"""
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_build_grid.py",
    "02_collect_data.py",
    "03_estimate.py",
    "04_optimize.py",
    "05_validate.py",
]


def main():
    for step in STEPS:
        print(f"\n===== {step} =====")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / step)])
        if result.returncode != 0:
            print(f"ОШИБКА на {step} (код {result.returncode}) — остановка пайплайна")
            sys.exit(result.returncode)
    print("\nПайплайн пройден полностью: grid → data → estimate → optimize → validate.")


if __name__ == "__main__":
    main()
