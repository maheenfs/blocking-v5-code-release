"""Shared plotting style loaded from ``plot_config.json``.

Keep visual defaults in the JSON file. Python code should only contain the
small amount of logic needed to apply those settings consistently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from experiment.system import configure_plot_environment

configure_plot_environment()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_CONFIG_PATH = Path(__file__).resolve().with_name("plot_config.json")

DEFAULT_STYLE_CONFIG: dict[str, object] = {
    "figure": {
        "figsize": [9.0, 5.4],
        "min_width": 10.5,
        "max_width": 12.5,
        "height": 5.4,
        "heatmap_figsize": [11.3, 6.6],
        "dpi": 200,
        "bbox_inches": "tight",
        "pad_inches": 0.08,
        "annotation_fontsize": 9,
    },
    "axes": {
        "ylabel": "Accuracy (%)",
        "grid_axis": "y",
        "grid_linestyle": ":",
        "grid_linewidth": 0.8,
        "grid_alpha": 0.45,
        "tick_labelsize": 9,
        "hide_spines": ["top", "right"],
    },
    "legend": {
        "fontsize": 9,
        "min_fontsize": 7.5,
        "frameon": False,
        "max_columns": 4,
        "loc": "upper center",
        "columnspacing": 1.25,
        "handlelength": 1.45,
        "handletextpad": 0.45,
        "labelspacing": 0.35,
    },
    "title": {
        "single_line_fontsize": 11.2,
        "multi_line_fontsize": 10.5,
        "single_line_pad": 14,
        "multi_line_pad": 18,
        "title_y": 0.985,
        "render": True,
    },
    "colors": {
        "palette": {
            "navy": "#4E6E8E",
            "blue": "#6B8BAF",
            "teal": "#4F8F8B",
            "deep_teal": "#3F7C85",
            "green": "#5E8C61",
            "red": "#B56B6B",
            "brick": "#8F5D56",
            "slate": "#7C8797",
            "purple": "#8F7AB8",
        },
        "methods": {
            "BL-AG": "#25B71A",
            "BL-AW": "#DB8070",
            "FT": "#346F89",
            "FT freeze_cnn": "#346F89",
            "FT no_freeze": "#25933D",
            "FT freeze_head": "#9AD4E0",
            "FT freeze_all": "#12A5AF",
            "MAX-SETB": "#8FB3D4",
            "NN-ANGLE": "#B88FB5",
            "RANDOM-SETB": "#8F5D56",
        },
        "freeze": {
            "no_freeze": "#25933D",
            "freeze_cnn": "#346F89",
            "freeze_head": "#9AD4E0",
            "freeze_all": "#12A5AF",
        },
        "blockages": {
            "0": "#4E6E8E",
            "20": "#6B8BAF",
            "40": "#4F8F8B",
            "60": "#5E8C61",
            "80": "#B56B6B",
            "100": "#8F5D56",
        },
        "beams": {
            "B0": "#4E6E8E",
            "B1": "#6B8BAF",
            "B2": "#4F8F8B",
            "B3": "#5E8C61",
            "B4": "#B56B6B",
            "B5": "#8F5D56",
            "B6": "#3F7C85",
            "B7": "#7C8797",
        },
        "kpis": {
            "primary": "#4E6E8E",
            "top1": "#6B8BAF",
            "top1_m1db": "#4F8F8B",
            "top2_m1db": "#5E8C61",
            "top3_m0db": "#B56B6B",
            "top3_m1db": "#4E6E8E",
            "top3_m2db": "#8F5D56",
        },
        "series_cycle": ["#4E6E8E", "#4F8F8B", "#B56B6B", "#5E8C61", "#6B8BAF", "#7C8797", "#3F7C85", "#8F5D56", "#8F7AB8"],
    },
    "overrides": {},
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def load_plot_style(path: str | Path = PLOT_CONFIG_PATH) -> dict[str, object]:
    """Load style JSON, falling back to safe defaults when the file is absent."""

    config = json.loads(json.dumps(DEFAULT_STYLE_CONFIG))
    path = Path(path)
    if not path.exists():
        return config
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Plot config must contain a JSON object: {path}")
    return _deep_merge(config, loaded)


STYLE_CONFIG = load_plot_style()


def _section(name: str) -> dict[str, object]:
    value = STYLE_CONFIG.get(name, {})
    return value if isinstance(value, dict) else {}


def _color_section(name: str) -> dict[str, str]:
    colors = _section("colors").get(name, {})
    if not isinstance(colors, dict):
        return {}
    return {str(key): str(value) for key, value in colors.items()}


PALETTE = _color_section("palette")
METHOD_COLORS = _color_section("methods")
FREEZE_COLORS = _color_section("freeze")
BLOCKAGE_COLORS = {int(key): value for key, value in _color_section("blockages").items()}
BEAM_ID_COLORS = _color_section("beams")
KPI_COLORS = _color_section("kpis")
SERIES_CYCLE = [str(value) for value in _section("colors").get("series_cycle", [])]
if not SERIES_CYCLE:
    SERIES_CYCLE = ["#4E6E8E", "#4F8F8B", "#B56B6B", "#5E8C61"]
PLOT_OVERRIDES = _section("overrides")

FIGURE_CONFIG = _section("figure")
FIGSIZE = tuple(float(v) for v in FIGURE_CONFIG.get("figsize", [9.0, 5.4]))  # type: ignore[arg-type]
FIG_W_MIN = float(FIGURE_CONFIG.get("min_width", FIGSIZE[0]))
FIG_H = float(FIGURE_CONFIG.get("height", FIGSIZE[1]))
DPI = int(FIGURE_CONFIG.get("dpi", 220))
ANNOTATION_FONTSIZE = int(FIGURE_CONFIG.get("annotation_fontsize", 9))


def slugify(value: object, default: str = "plot") -> str:
    import re

    text = str(value or default).strip()
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def figure_size(kind: str = "default") -> tuple[float, float]:
    key = "heatmap_figsize" if kind == "heatmap" else "figsize"
    raw = FIGURE_CONFIG.get(key, FIGSIZE)
    return tuple(float(v) for v in raw)  # type: ignore[arg-type]


def dynamic_figure_size(label_count: int, series_count: int = 1, *, kind: str = "default") -> tuple[float, float]:
    if kind == "heatmap":
        return figure_size("heatmap")
    raw_width = 1.5 * max(1, int(label_count)) + 0.6 * max(1, int(series_count))
    width = max(FIG_W_MIN, raw_width)
    if "max_width" in FIGURE_CONFIG:
        width = min(width, float(FIGURE_CONFIG["max_width"]))
    return (width, FIG_H)


def color_for(label: object, index: int = 0) -> str:
    text = str(label)
    if text in METHOD_COLORS:
        return METHOD_COLORS[text]
    if text in FREEZE_COLORS:
        return FREEZE_COLORS[text]
    if text in BEAM_ID_COLORS:
        return BEAM_ID_COLORS[text]
    if text in KPI_COLORS:
        return KPI_COLORS[text]
    return SERIES_CYCLE[index % len(SERIES_CYCLE)]


def plot_override(plot_id: object | None) -> dict[str, dict[str, object]]:
    if plot_id is None:
        return {}
    raw = str(plot_id).replace("\\", "/")
    candidates = [raw, Path(raw).name]
    if "plots/selected_plots/" in raw:
        candidates.append(raw.split("plots/selected_plots/", 1)[1])
    for candidate in candidates:
        override = PLOT_OVERRIDES.get(candidate)
        if isinstance(override, dict):
            return override  # type: ignore[return-value]
    return {}


def smart_ylim(values: Iterable[float], *, floor_zero: bool = False) -> tuple[float, float]:
    finite = [float(v) for v in values if v == v]
    if not finite:
        return (0.0, 100.0)
    lo = min(finite)
    hi = max(finite)
    if lo == hi:
        lo -= 5.0
        hi += 5.0
    span = max(20.0, hi - lo)
    pad = max(2.0, span * 0.08)
    ymin = lo - pad
    ymax = hi + pad
    if floor_zero:
        ymin = min(0.0, ymin)
    ymin = max(0.0, 5.0 * int(ymin // 5))
    ymax = min(105.0, 5.0 * int((ymax + 4.999) // 5))
    if ymax - ymin < 20.0:
        ymax = min(105.0, ymin + 20.0)
    return ymin, ymax


def legend_kwargs(series_count: int, *, plot_id: object | None = None) -> dict[str, object]:
    legend = _section("legend")
    override = plot_override(plot_id).get("legend", {})
    config = _deep_merge(legend, override) if isinstance(override, dict) else legend
    return {
        "fontsize": float(config.get("fontsize", 8.5)),
        "frameon": bool(config.get("frameon", False)),
        "ncols": min(int(config.get("max_columns", 3)), max(1, int(series_count))),
        "loc": str(config.get("loc", "best")),
    }


def clean_legend_label(label: str) -> str:
    text = " ".join(str(label).strip().split())
    replacements = {
        "FT freeze_cnn": "FT freeze-CNN",
        "FT no_freeze": "FT no-freeze",
        "FT freeze_head": "FT frozen-head",
        "FT freeze_all": "FT frozen-all",
        "FT:freeze_cnn": "FT freeze-CNN",
        "freeze_cnn": "freeze-CNN",
        "no_freeze": "no-freeze",
        "freeze_head": "frozen-head",
        "freeze_all": "frozen-all",
        "Non-ML: MAX-SETB": "MAX-SETB",
        "Non-ML: NN-ANGLE": "NN-ANGLE",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def clean_plot_title(title: str) -> str:
    text = "\n".join(" ".join(line.strip().split()) for line in str(title).splitlines() if line.strip())
    replacements = {
        "accuracy vs blockage": "Accuracy vs blockage",
        "accuracy vs train fraction": "Accuracy vs training fraction",
        "train fraction": "training fraction",
        "epoch budget": "epoch-budget",
        "freeze_cnn": "freeze-CNN",
        "no_freeze": "no-freeze",
        "freeze_head": "frozen-head",
        "freeze_all": "frozen-all",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def legend_column_count(labels: list[str], requested_ncol: int | None = None) -> int:
    count = len(labels)
    if count <= 0:
        return 1
    if requested_ncol is not None:
        return max(1, min(count, int(requested_ncol)))
    longest = max(len(label) for label in labels)
    if count <= 3:
        return count
    if count <= 5:
        return min(count, 5 if longest <= 18 else 3)
    if count <= 8:
        return 4 if longest <= 22 else 3
    return 4


def set_plot_title(ax: plt.Axes, title: str, *, plot_id: object | None = None) -> None:
    title_cfg = _section("title")
    title_override = plot_override(plot_id).get("title", {})
    if isinstance(title_override, dict):
        title_cfg = _deep_merge(title_cfg, title_override)
    if not bool(title_cfg.get("render", True)):
        ax.set_title("")
        return
    text = clean_plot_title(str(title_cfg.get("text", title)))
    multiline = "\n" in text
    fontsize = float(title_cfg.get("multi_line_fontsize" if multiline or len(text) > 90 else "single_line_fontsize", 11.2))
    pad = float(title_cfg.get("multi_line_pad" if multiline else "single_line_pad", 14))
    ax.set_title(text, fontsize=fontsize, pad=pad)


def top_band_layout(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    handles: list[object] | None = None,
    labels: list[str] | None = None,
    ncol: int | None = None,
    plot_id: object | None = None,
) -> None:
    legend_cfg = _section("legend")
    legend_override = plot_override(plot_id).get("legend", {})
    if isinstance(legend_override, dict):
        legend_cfg = _deep_merge(legend_cfg, legend_override)
    title_cfg = _section("title")
    title_override = plot_override(plot_id).get("title", {})
    if isinstance(title_override, dict):
        title_cfg = _deep_merge(title_cfg, title_override)

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    handles = list(handles)
    labels = [clean_legend_label(label) for label in list(labels)]
    title = clean_plot_title(ax.get_title())
    ax.set_title("")
    for existing in list(fig.legends):
        existing.remove()

    title_lines = max(1, title.count("\n") + 1) if title else 0
    title_fs = float(title_cfg.get("multi_line_fontsize" if title_lines > 1 or len(title) > 92 else "single_line_fontsize", 11.2))
    requested_ncol = ncol if ncol is not None else legend_cfg.get("ncol")
    selected_ncol = legend_column_count(labels, requested_ncol=int(requested_ncol) if requested_ncol is not None else None)
    legend_rows = int(math.ceil(len(labels) / selected_ncol)) if labels else 0
    title_y = float(title_cfg.get("title_y", 0.985)) if title else 0.965
    legend_y = 0.925 - 0.035 * max(0, title_lines - 1) if title else 0.965
    axes_top = 0.96 if not labels else legend_y - 0.052 * max(1, legend_rows) - 0.02
    axes_top = max(0.54, min(0.90, axes_top))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, axes_top))
    if title:
        fig.suptitle(title, fontsize=title_fs, y=title_y)
    if not labels:
        return
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        borderaxespad=0.0,
        fontsize=float(legend_cfg.get("fontsize", 9)),
        ncol=selected_ncol,
        frameon=bool(legend_cfg.get("frameon", False)),
        columnspacing=float(legend_cfg.get("columnspacing", 1.25)),
        handlelength=float(legend_cfg.get("handlelength", 1.45)),
        handletextpad=float(legend_cfg.get("handletextpad", 0.45)),
        labelspacing=float(legend_cfg.get("labelspacing", 0.35)),
    )
    if legend.get_frame() is not None:
        legend.get_frame().set_alpha(0.0 if not bool(legend_cfg.get("frameon", False)) else 0.92)


def style_axes(ax: plt.Axes, *, ylabel: str = "Accuracy (%)", title: str = "", plot_id: object | None = None) -> None:
    axes = _section("axes")
    override = plot_override(plot_id)
    ax.set_ylabel(ylabel or str(axes.get("ylabel", "Accuracy (%)")))
    if title:
        set_plot_title(ax, title, plot_id=plot_id)
    ax.grid(
        axis=str(axes.get("grid_axis", "y")),
        linestyle=str(axes.get("grid_linestyle", ":")),
        linewidth=float(axes.get("grid_linewidth", 0.8)),
        alpha=float(axes.get("grid_alpha", 0.45)),
    )
    for spine in axes.get("hide_spines", ["top", "right"]):  # type: ignore[assignment]
        if str(spine) in ax.spines:
            ax.spines[str(spine)].set_visible(False)
    ax.tick_params(axis="both", labelsize=float(axes.get("tick_labelsize", 9)))
    ax.set_axisbelow(True)

    axes_override = override.get("axes", {})
    if "ylim" in axes_override:
        ax.set_ylim(*axes_override["ylim"])  # type: ignore[arg-type]
    if "xlim" in axes_override:
        ax.set_xlim(*axes_override["xlim"])  # type: ignore[arg-type]
    if "ylabel" in axes_override:
        ax.set_ylabel(str(axes_override["ylabel"]))
    if "xlabel" in axes_override:
        ax.set_xlabel(str(axes_override["xlabel"]))


def save_figure(fig: plt.Figure, path: Path, *, plot_id: object | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    override = plot_override(plot_id)
    save_override = override.get("save", {})
    dpi = int(save_override.get("dpi", DPI))
    bbox = str(save_override.get("bbox_inches", FIGURE_CONFIG.get("bbox_inches", "tight")))
    pad_inches = float(save_override.get("pad_inches", FIGURE_CONFIG.get("pad_inches", 0.08)))
    fig.savefig(path, dpi=dpi, bbox_inches=bbox, pad_inches=pad_inches)
    plt.close(fig)
    return path
