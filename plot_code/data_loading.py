"""CSV and result-tree loading for plot_code."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Iterable

from .config import DEFAULT_RESULTS_ROOT, load_config, normalize_beam_token


@dataclass(frozen=True)
class RunInfo:
    path: Path
    seed: int | None
    pattern: int | None
    beam_index: int | None
    beam: str
    rows: list[dict[str, str]]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_run_name(path: Path) -> tuple[int | None, int | None, int | None]:
    match = re.search(r"seed(\d+)_P(\d+)_blockB(\d+)", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match = re.search(r"_P(\d+)_blockB(\d+)", path.name)
    if match:
        return None, int(match.group(1)), int(match.group(2))
    return None, None, None


def discover_runs(results_root: str | Path = DEFAULT_RESULTS_ROOT) -> list[RunInfo]:
    root = Path(results_root)
    runs: list[RunInfo] = []
    if not root.exists():
        return runs
    for path in sorted(root.glob("blocking_v5*_P*_blockB*")):
        if not path.is_dir():
            continue
        rows = read_csv(path / "results.csv")
        if not rows:
            continue
        seed, pattern, beam_index = parse_run_name(path)
        runs.append(
            RunInfo(
                path=path,
                seed=seed,
                pattern=pattern,
                beam_index=beam_index,
                beam=normalize_beam_token(beam_index if beam_index is not None else ""),
                rows=rows,
            )
        )
    return runs


def filtered_rows(
    rows: Iterable[dict[str, str]],
    *,
    method: str | None = None,
    freeze: str | None = None,
    train_frac: float | None = None,
    blockage: int | None = None,
    epochs: int | None = None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if freeze is not None and row.get("freeze") != freeze:
            continue
        if train_frac is not None and abs(as_float(row.get("train_frac")) - float(train_frac)) > 1e-8:
            continue
        if blockage is not None and as_int(row.get("blockage_%")) != int(blockage):
            continue
        if epochs is not None and as_int(row.get("ft_epochs")) != int(epochs):
            continue
        out.append(row)
    return out


def row_value(row: dict[str, str], column: str) -> float:
    return as_float(row.get(column))


def mean_value(rows: Iterable[dict[str, str]], column: str) -> float:
    values = [row_value(row, column) for row in rows]
    values = [value for value in values if value == value]
    return mean(values) if values else float("nan")


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if value == value]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return mean(finite), stdev(finite)


def config_at(path: str | Path | None) -> dict:
    return load_config(path) if path else {}
