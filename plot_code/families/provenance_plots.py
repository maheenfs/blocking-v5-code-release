"""Provenance and Set-B mapping plots."""

from __future__ import annotations

from pathlib import Path

from ..charts import grouped_bar
from ..config import DEFAULT_CONFIG_PATH, DEFAULT_PLOTS_ROOT, DEFAULT_RESULTS_ROOT, PlotSelection, load_config
from ..data_loading import as_float, discover_runs, read_csv
from ..filters import canonical_family, family_enabled
from ..organization import PlotOutput, organize_results


def build_provenance_plots(
    *,
    config: str | Path = DEFAULT_CONFIG_PATH,
    out_dir: str | Path | None = None,
    selection: PlotSelection | None = None,
    organize: bool = False,
) -> list[PlotOutput]:
    selection = selection or PlotSelection()
    if not family_enabled(selection, "provenance", "setup"):
        return []
    output_family = "setup" if any(canonical_family(item) == "setup" for item in selection.families) else "provenance"
    cfg = load_config(config)
    output_root = Path(out_dir) if out_dir is not None else DEFAULT_PLOTS_ROOT / "provenance"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[PlotOutput] = []
    for pattern in (1, 2):
        mapping = None
        for run in discover_runs(cfg.get("outdir", DEFAULT_RESULTS_ROOT)):
            if run.pattern == pattern and (run.path / "setb_mapping.csv").exists():
                mapping = read_csv(run.path / "setb_mapping.csv")
                break
        if not mapping:
            continue
        labels = [row.get("setb_beam", row.get("setb_index", f"B{i}")) for i, row in enumerate(mapping)]
        values = [as_float(row.get("tx_index", row.get("seta_index", i))) for i, row in enumerate(mapping)]
        path = grouped_bar(
            labels=labels,
            series=[("Set-A index", values, None)],
            title=f"Pattern {pattern}: Set-B to Set-A beam mapping",
            ylabel="Set-A transmitter index",
            path=output_root / f"provenance_p{pattern}_beam_mapping.png",
        )
        outputs.append(
            PlotOutput(
                source_path=path,
                title=f"Pattern {pattern}: Set-B to Set-A beam mapping",
                kpi="none",
                family=output_family,
                scope="results",
                pattern=pattern,
                methods=("provenance",),
            )
        )
    if organize or selection.organize:
        organize_results(cfg.get("outdir", DEFAULT_RESULTS_ROOT), selection=selection, copy_mode=selection.copy_mode, clean=selection.clean_organized, plot_outputs=outputs)
    return outputs
