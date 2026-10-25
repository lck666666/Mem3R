"""Make the bundled model packages available to standalone entry points."""

import sys
from pathlib import Path


def add_path_to_dust3r(checkpoint_path=None):
    """Use this checkout's sources, regardless of the checkpoint's location.

    ``checkpoint_path`` is retained for compatibility with existing launchers.
    A checkpoint contains weights and configuration, not executable packages.
    """
    src = str(Path(__file__).resolve().parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
