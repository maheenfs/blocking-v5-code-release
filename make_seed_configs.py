"""Generate seed-specific Blocking V5 config files.

Each generated config changes both ``training.seed`` and ``run_name_prefix`` so
multi-seed runs can coexist in the same results folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create seed-specific Blocking V5 config files.")
    parser.add_argument("--base", default=str(Path(__file__).with_name("config.json")), help="Base config JSON")
    parser.add_argument("--seeds", default="123,456,789", help="Comma-separated seeds")
    parser.add_argument("--outdir", default=str(Path(__file__).with_name("seed_configs")), help="Output directory")
    parser.add_argument("--prefix-base", default="", help="Optional base run_name_prefix; defaults to config value")
    args = parser.parse_args()

    base_path = Path(args.base).expanduser().resolve()
    payload = load_json(base_path)
    prefix_base = str(args.prefix_base or payload.get("run_name_prefix", "blocking_v5"))
    outdir = Path(args.outdir).expanduser().resolve()

    for seed in parse_int_list(args.seeds):
        cfg = json.loads(json.dumps(payload))
        cfg.setdefault("training", {})
        cfg["training"]["seed"] = int(seed)
        cfg["training"].pop("seeds", None)
        cfg["run_name_prefix"] = f"{prefix_base}_seed{int(seed)}"
        out_path = outdir / f"config_seed_{int(seed)}.json"
        write_json(out_path, cfg)
        print(out_path)


if __name__ == "__main__":
    main()
