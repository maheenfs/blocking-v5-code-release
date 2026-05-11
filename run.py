"""Command-line entrypoint for Blocking V5 release experiments."""

import sys

from experiment.preflight import ConfigSelectionError, MissingDependencyError
from runner.cli import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except (ConfigSelectionError, MissingDependencyError, FileNotFoundError, ValueError) as exc:
        print(f"Run setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
