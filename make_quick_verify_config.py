"""Create a small FT verification config that reuses packaged BL-AG weights."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
from pathlib import Path


STORED_BL_AG = Path("stored_weights") / "baseline_pretrained" / "BL-AG_clean_pretrain_weights.pt"


def parse_run_name(run_name: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"blocking_v5_seed(\d+)_P(\d+)_blockB(\d+)", run_name)
    if match is None:
        raise SystemExit(
            "run name must look like blocking_v5_seed123_P1_blockB1; "
            f"received {run_name!r}"
        )
    seed, pattern, blocked_beam = (int(value) for value in match.groups())
    return seed, pattern, blocked_beam


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"verification config already exists: {path}\n"
            "Pass --overwrite to replace it, or choose --label/--output-config."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_output_label(args: argparse.Namespace) -> str:
    raw_label = (
        f"{args.run_name}_blockage{args.blockage}_"
        f"frac{float_token(float(args.train_frac))}_{args.freeze_mode}_"
        f"epochs{args.epochs}"
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label).strip("_")


def resolve_output_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    label = args.label or build_output_label(args)
    outdir = args.outdir or Path("quick_verify_results") / label
    output_config = args.output_config or Path("quick_verify_configs") / f"{label}.json"
    return label, output_config, outdir


def build_config(args: argparse.Namespace) -> dict:
    seed, pattern, blocked_beam = parse_run_name(args.run_name)
    config = load_json(args.config)
    training = dict(config["training"])
    plotting = dict(config.get("plotting", {}))

    training["seed"] = seed
    training["seeds"] = []
    training["train_fracs_sweep"] = [float(args.train_frac)]
    training["ft_freeze_modes"] = [str(args.freeze_mode)]
    training["ft_epoch_sweep"] = [int(args.epochs)]

    plotting["runner_postprocessing"] = {"after_each_run": [], "after_batch": []}

    config["outdir"] = str(args.outdir)
    config["stages"] = ["prepare", "finetune", "aggregate"]
    config["experiments"] = ["FT"]
    config["blockage_levels"] = [int(args.blockage)]
    config["batch_jobs"] = [
        {
            "pattern": pattern,
            "blocked_beam_index": blocked_beam,
            "description": f"quick verification P{pattern} block B{blocked_beam}",
        }
    ]
    config["training"] = training
    config["plotting"] = plotting
    return config


def copy_stored_weight(run_name: str, outdir: Path, overwrite: bool) -> Path:
    source = Path("results") / run_name / STORED_BL_AG
    target = outdir / run_name / STORED_BL_AG
    if not source.exists():
        raise SystemExit(f"stored BL-AG weight is missing: {source}")
    if target.exists() and not overwrite:
        raise SystemExit(
            f"stored BL-AG weight already exists in verification output: {target}\n"
            "Pass --overwrite to replace it, or choose --label/--outdir."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def guard_run_output(run_name: str, outdir: Path, overwrite: bool) -> None:
    run_dir = outdir / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"verification run output already exists: {run_dir}\n"
            "Pass --overwrite to replace it, or choose --label/--outdir."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated FT-only verification config and copy the "
            "packaged BL-AG weight needed for that check."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Default: quick_verify_configs/<setting-label>.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Default: quick_verify_results/<setting-label>",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label used in default config/result paths.",
    )
    parser.add_argument("--run-name", default="blocking_v5_seed123_P1_blockB1")
    parser.add_argument("--blockage", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.01)
    parser.add_argument("--freeze-mode", default="freeze_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing verification config/output for the same setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label, output_config, outdir = resolve_output_paths(args)
    args.outdir = outdir
    guard_run_output(args.run_name, outdir, args.overwrite)
    config = build_config(args)
    write_json(output_config, config, args.overwrite)
    copied_weight = copy_stored_weight(args.run_name, outdir, args.overwrite)
    run_result = outdir / args.run_name / "results.csv"
    command = [
        "python3",
        "-B",
        "run.py",
        "--config",
        str(output_config),
        "--run-name",
        args.run_name,
        "--no-dashboard",
    ]
    print(f"Verification label: {label}")
    print(f"Wrote config: {output_config}")
    print(f"Results root: {outdir}")
    print(f"Copied stored BL-AG weight: {copied_weight}")
    print(f"Expected result CSV after run: {run_result}")
    print("Run:")
    print(" ".join(shlex.quote(part) for part in command))


if __name__ == "__main__":
    main()
