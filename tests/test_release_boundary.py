"""Checks that the release package is standalone and does not use legacy entrypoints internally."""

from __future__ import annotations

from pathlib import Path
import unittest


RELEASE_ROOT = Path(__file__).resolve().parents[1]


class ReleaseBoundaryTests(unittest.TestCase):
    def test_internal_python_files_do_not_import_flat_core_or_plots(self) -> None:
        bad_hits: list[str] = []
        allowed = {RELEASE_ROOT / "core.py", RELEASE_ROOT / "plots.py"}
        for path in RELEASE_ROOT.rglob("*.py"):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            legacy_tokens = (
                "from " + "core import",
                "import " + "core",
                "from " + "plots import",
                "import " + "plots",
            )
            for token in legacy_tokens:
                if token in text:
                    bad_hits.append(f"{path.relative_to(RELEASE_ROOT)}: {token}")

        self.assertEqual(bad_hits, [])

    def test_release_code_does_not_reference_local_machine_or_old_project_paths(self) -> None:
        bad_hits: list[str] = []
        forbidden = (
            "/" + "Users/",
            "Desktop/" + "Blocking_Final",
            "SESSION" + "_NOTES",
            "../" + "blocking_v5",
            "../../" + "blocking_v5",
            "blocking_v5/" + "plots.py",
        )
        generated_dirs = {"results", "__pycache__"}
        for path in RELEASE_ROOT.rglob("*"):
            parts = path.relative_to(RELEASE_ROOT).parts
            if any(part.startswith(".") for part in parts):
                continue
            if any(part in generated_dirs for part in parts):
                continue
            if not path.is_file() or path.suffix not in {".py", ".md", ".json", ".csv"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    bad_hits.append(f"{path.relative_to(RELEASE_ROOT)}: {token}")

        self.assertEqual(bad_hits, [])


if __name__ == "__main__":
    unittest.main()
