"""Configuration helpers for plotting commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "results"
DEFAULT_PLOTS_ROOT = PACKAGE_ROOT / "plots"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.json"
DEFAULT_SELECTED_ROOT = DEFAULT_PLOTS_ROOT / "selected_plots"
DEFAULT_ORGANIZED_ROOT = DEFAULT_PLOTS_ROOT / "organized_plots"
DEFAULT_PLOT_CONFIG_PATH = Path(__file__).resolve().with_name("plot_config.json")


@dataclass(frozen=True)
class PlotSelection:
    """CLI-selected plotting scope from flags."""

    kpis: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    patterns: tuple[int, ...] = ()
    blocked_beams: tuple[str, ...] = ()
    train_fracs: tuple[float, ...] = ()
    blockages: tuple[int, ...] = ()
    epochs: tuple[int, ...] = ()
    organize: bool = False
    clean_organized: bool = False
    copy_mode: str = "copy"

def parse_csv_text(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_int_csv(value: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in parse_csv_text(value))


def parse_float_csv(value: str | None) -> tuple[float, ...]:
    return tuple(float(part) for part in parse_csv_text(value))


def normalize_beam_token(value: object) -> str:
    text = str(value).strip().upper()
    if not text:
        return text
    return text if text.startswith("B") else f"B{text}"


def normalize_family_token(value: object) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
