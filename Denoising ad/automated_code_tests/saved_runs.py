"""
Helper for tests that read experiment runs saved on the computer that wrote them.

The results/ folder is not stored in Git, so a fresh copy of the project has
no result13, result14, ... folders. Tests that need one of those runs are
skipped there, with a message saying which run is missing, instead of failing.
"""

import unittest
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def needs_saved_runs(*names):
    """Skips a test or test class unless these saved runs exist locally."""
    missing = [name for name in names if not (RESULTS_DIR / name / "run_config.json").is_file()]
    return unittest.skipIf(
        bool(missing),
        f"needs the locally saved run(s) {', '.join(missing)} in results/ (not stored in Git)",
    )
