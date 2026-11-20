"""
Blank the server coordinates (longitude, latitude) in every CodeCarbon ``emissions*.csv``
under the given directories, using ``src.pipeline.scrub_location``.

Usage: python scripts/scrub_emissions_location.py results reproduction outputs mlruns
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from src.pipeline import scrub_location


def main() -> None:
    """Scrub all logs and report counts."""
    files = [fp for root in sys.argv[1:] for fp in sorted(Path(root).rglob("emissions*.csv"))]
    changed = {fp: scrub_location(fp) for fp in files}
    print(f"{len(files)} files scanned, {sum(1 for n in changed.values() if n)} changed, "
          f"{sum(changed.values())} rows scrubbed")


if __name__ == "__main__":
    main()
