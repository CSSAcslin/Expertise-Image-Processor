"""Launch the Web workbench from a checkout."""
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from eip.web.main import main
    raise SystemExit(main())
