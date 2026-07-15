"""Точка входа: графики для отчёта → outputs/figures/ (валидация, wealth, скоры)."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from clinic_siting.optimize.scores import equity_score, profit_score
from clinic_siting.utils.io import load_config, load_cells

_GREEN, _AMBER, _GREY = "#0F6E56", "#BA7517", "#9aa0a6"


def _fig_validation(figures_dir: Path, tables_dir: str):
    rep = pd.read_csv(f"{tables_dir}/validation.csv")
    model = rep[rep.scenario == "model"].iloc[0]
    base = rep[rep.scenario.str.startswith("baseline")].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    vals = [model["before"], model["after"], base["after"]]
    bars = ax1.bar(["До", "После\n(модель)", "После\n(baseline)"], vals, color=[_GREY, _GREEN, _AMBER])
    ax1.set_ylabel("Среднее время до клиники, мин")
    ax1.set_title("Доступность до и после")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    imp = [model["improvement"], base["improvement"]]
    bars2 = ax2.bar(["Модель", "Baseline"], imp, color=[_GREEN, _AMBER])
    ax2.set_ylabel("Сокращение времени, мин")
    ax2.set_title("Улучшение: модель vs baseline")
    for b, v in zip(bars2, imp):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "validation_access.png", dpi=130)
    plt.close(fig)


def _fig_wealth(figures_dir: Path, cells):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cells["wealth_score"].plot(kind="hist", bins=range(0, 12), ax=ax1, color=_GREEN, edgecolor="white", rwidth=0.9)
    ax1.set_xlabel("wealth_score")
    ax1.set_ylabel("Клеток")
    ax1.set_title("Распределение wealth_score")

    grp = cells.groupby(cells["existing_clinics"].clip(upper=5))["wealth_score"].mean()
    ax2.plot(grp.index, grp.values, marker="o", color=_GREEN)
    ax2.set_xlabel("Клиник рядом (обрезано до 5)")
    ax2.set_ylabel("Средний wealth_score")
    ax2.set_title("wealth_score vs число клиник")

    plt.tight_layout()
    plt.savefig(figures_dir / "wealth_scores.png", dpi=130)
    plt.close(fig)


def _fig_scores(figures_dir: Path, cells):
    equity = equity_score(cells)
    profit = profit_score(cells)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(equity, bins=20, alpha=0.6, label="equity (→ гос)", color=_GREEN)
    ax.hist(profit, bins=20, alpha=0.6, label="profit (→ частная)", color=_AMBER)
    ax.set_xlabel("Скор (0-1)")
    ax.set_ylabel("Клеток")
    ax.set_title("Распределение скоров equity/profit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "score_distribution.png", dpi=130)
    plt.close(fig)


def main():
    cfg = load_config()
    figures_dir = Path(cfg["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(f"{cfg['output']['processed_dir']}/cells_estimated.parquet")

    _fig_validation(figures_dir, cfg["output"]["tables_dir"])
    _fig_wealth(figures_dir, cells)
    _fig_scores(figures_dir, cells)

    print(f"Графики сохранены в {figures_dir}: validation_access.png, wealth_scores.png, score_distribution.png")


if __name__ == "__main__":
    main()
