"""KPI naming and metric selection helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    column: str


def primary_metric_key(config: dict | None = None) -> str:
    if not config:
        return "top3_m1db"
    kpi = dict(config.get("kpi", {}))
    topk = int(kpi.get("primary_topk", 3))
    margin = int(kpi.get("primary_margin_db", 1))
    return f"top{topk}_m{margin}db"


def primary_metric_label(config: dict | None = None) -> str:
    key = primary_metric_key(config)
    return metric_label(key)


def metric_label(key: str) -> str:
    labels = {
        "primary": "Top-3 within 1 dB accuracy (%)",
        "test_acc": "Exact class accuracy (%)",
        "top1": "Top-1 exact accuracy (%)",
        "top2": "Top-2 inclusion accuracy (%)",
        "top3": "Top-3 inclusion accuracy (%)",
        "top4": "Top-4 inclusion accuracy (%)",
        "top1_m0db": "Top-1 within 0 dB accuracy (%)",
        "top1_m1db": "Top-1 within 1 dB accuracy (%)",
        "top2_m1db": "Top-2 within 1 dB accuracy (%)",
        "top3_m0db": "Top-3 within 0 dB accuracy (%)",
        "top3_m1db": "Top-3 within 1 dB accuracy (%)",
        "top3_m2db": "Top-3 within 2 dB accuracy (%)",
    }
    return labels.get(key, f"{key.replace('_', ' ')} (%)")


def metric_column(key: str, config: dict | None = None) -> str:
    if key == "primary":
        key = primary_metric_key(config)
    if key == "test_acc":
        return "test_acc_%"
    return f"test_{key}_%"


def metric_specs_from_keys(keys: tuple[str, ...] | list[str] | None, config: dict | None = None) -> list[MetricSpec]:
    requested = list(keys or ["primary"])
    specs: list[MetricSpec] = []
    for key in requested:
        specs.append(MetricSpec(key=key, label=metric_label(key if key != "primary" else primary_metric_key(config)), column=metric_column(key, config)))
    return specs


def metric_specs_from_config(config: dict | None = None) -> list[MetricSpec]:
    plotting = dict((config or {}).get("plotting", {}))
    keys = plotting.get("metric_plot_keys", ["primary"])
    if not isinstance(keys, list):
        keys = [keys]
    return metric_specs_from_keys(tuple(str(key) for key in keys), config)


def multi_kpi_specs(config: dict | None = None) -> list[MetricSpec]:
    plotting = dict((config or {}).get("plotting", {}))
    keys = plotting.get("multi_kpi_metric_keys", ["top1", "top1_m1db", "top2_m1db", "top3_m0db", "primary", "top3_m2db"])
    if not isinstance(keys, list):
        keys = [keys]
    return metric_specs_from_keys(tuple(str(key) for key in keys), config)
