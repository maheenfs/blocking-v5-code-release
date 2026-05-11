"""Small chart builders used by plot families."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .style import ANNOTATION_FONTSIZE, color_for, dynamic_figure_size, figure_size, save_figure, smart_ylim, style_axes, top_band_layout, plt


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _format_value(value: float, value_format: str) -> str:
    return value_format.format(float(value))


def _annotate_bar(ax, rect, value: float, *, value_format: str = "{:.1f}") -> tuple[object, float, float, str] | None:
    if not _finite(value):
        return None
    x_value = rect.get_x() + rect.get_width() / 2.0
    y_value = float(rect.get_height())
    text = _format_value(float(value), value_format)
    annotation = ax.annotate(
        text,
        xy=(x_value, y_value),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
    )
    return annotation, float(x_value), y_value, text


def _merge_repeated_labels(ax, records: list[tuple[object, float, float, str]], *, offset_points: int) -> None:
    """Show one centered label when same-height marks share one x tick."""

    groups: dict[tuple[int, str], list[tuple[object, float, float, str]]] = {}
    for record in records:
        _, x_value, _, text = record
        groups.setdefault((round(float(x_value)), text), []).append(record)

    for (_, text), group in groups.items():
        if len(group) < 2:
            continue
        for annotation, _, _, _ in group:
            annotation.set_visible(False)
        x_values = [item[1] for item in group]
        y_value = max(item[2] for item in group)
        ax.annotate(
            text,
            xy=(float(sum(x_values) / len(x_values)), float(y_value)),
            xytext=(0, offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )


def grouped_bar(
    *,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], Sequence[float] | None]],
    title: str,
    ylabel: str,
    path: Path,
    plot_id: object | None = None,
    ylim: tuple[float, float] | None = None,
    annotate: bool = True,
    value_format: str = "{:.1f}",
) -> Path:
    fig, ax = plt.subplots(figsize=dynamic_figure_size(len(labels), len(series)))
    x = np.arange(len(labels), dtype=float)
    series_values = [(name, [float(value) for value in values]) for name, values, _ in series]
    count = max(1, len(series))
    width = min(0.8 / count, 0.22)
    offsets = (np.arange(count) - (count - 1) / 2.0) * width
    all_values: list[float] = []
    label_records: list[tuple[object, float, float, str]] = []
    for idx, ((name, y), (_, _, errors)) in enumerate(zip(series_values, series)):
        all_values.extend(y)
        err = None if errors is None else [float(value) for value in errors]
        bars = ax.bar(
            x + offsets[idx],
            y,
            width=width,
            yerr=err,
            capsize=3 if err is not None else 0,
            label=name,
            color=color_for(name, idx),
            edgecolor="white",
            linewidth=0.6,
        )
        if annotate:
            for rect, value in zip(bars, y):
                record = _annotate_bar(ax, rect, value, value_format=value_format)
                if record is not None:
                    label_records.append(record)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(*(ylim or smart_ylim(all_values)))
    _merge_repeated_labels(ax, label_records, offset_points=4)
    style_axes(ax, ylabel=ylabel, title=title, plot_id=plot_id)
    if len(series) > 1:
        top_band_layout(fig, ax, plot_id=plot_id)
    else:
        top_band_layout(fig, ax, handles=[], labels=[], plot_id=plot_id)
    return save_figure(fig, path, plot_id=plot_id)


def categorical_bar(
    *,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    title: str,
    ylabel: str,
    path: Path,
    plot_id: object | None = None,
    ylim: tuple[float, float] | None = None,
    annotate: bool = True,
    value_format: str = "{:.1f}",
) -> Path:
    """Render a multi-series categorical relationship as grouped bars."""

    return grouped_bar(
        labels=labels,
        series=[(name, values, None) for name, values in series],
        title=title,
        ylabel=ylabel,
        path=path,
        plot_id=plot_id,
        ylim=ylim,
        annotate=annotate,
        value_format=value_format,
    )


def heatmap(
    *,
    matrix: Sequence[Sequence[float]],
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    path: Path,
    plot_id: object | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=figure_size("heatmap"))
    data = np.array(matrix, dtype=float)
    finite_values = data[np.isfinite(data)]
    if finite_values.size:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    else:
        vmin, vmax = 0.0, 1.0
    image = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title, fontsize=11, pad=12)
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if value == value:
                color = "white" if float(value) <= (vmin + vmax) / 2.0 else "#111111"
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    top_band_layout(fig, ax, handles=[], labels=[], plot_id=plot_id)
    return save_figure(fig, path, plot_id=plot_id)
