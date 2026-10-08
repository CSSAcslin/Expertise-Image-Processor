"""Launch from a checkout without installing the package."""
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from eip.__main__ import desktop_main
    raise SystemExit(desktop_main())
