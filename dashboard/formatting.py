"""Formatting helpers shared by dashboard modules."""

from __future__ import annotations

import html
from typing import Sequence


def format_seconds(seconds: object) -> str:
    """Format seconds as m:ss or h:mm:ss; return n/a for missing values."""
    try:
        total = max(0.0, float(seconds))
    except Exception:
        return "n/a"
    hours = int(total // 3600)
    total -= hours * 3600
    minutes = int(total // 60)
    seconds_int = int(round(total - minutes * 60))
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds_int:02d}"
    return f"{minutes:d}:{seconds_int:02d}"


def format_runtime_float(value: object, *, ndigits: int = 1) -> str:
    """Format optional dashboard numbers without raising on missing values."""
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if not number == number:
        return "n/a"
    return f"{number:.{int(ndigits)}f}"


def format_optional_float(value: object, *, ndigits: int = 3, suffix: str = "") -> str:
    """Format optional training metrics for compact dashboard tables."""
    try:
        number = float(value)
    except Exception:
        return ""
    if not number == number:
        return ""
    return f"{number:.{int(ndigits)}f}{suffix}"


def render_html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render one plain HTML table with escaped cell values."""
    head_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_html = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body_html.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_html)}</tbody></table>"


def format_list(values: object, *, suffix: str = "") -> str:
    """Format a config list in a way that stays readable in narrow tables."""
    if values in (None, ""):
        return "default"
    if isinstance(values, str):
        return values
    if not isinstance(values, (list, tuple, set)):
        return str(values)
    rendered = [f"{value}{suffix}" for value in values]
    return ", ".join(str(value) for value in rendered) if rendered else "none"


def format_train_fraction(value: object) -> str:
    """Show fractions as percentages when the value is between 0 and 1."""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if 0.0 <= number <= 1.0:
        return f"{100.0 * number:g}%"
    return f"{number:g}"


def format_train_fraction_list(values: object) -> str:
    """Format the configured FT adaptation fractions."""
    if values in (None, ""):
        return "default"
    if not isinstance(values, (list, tuple, set)):
        return str(values)
    return ", ".join(format_train_fraction(value) for value in values)
