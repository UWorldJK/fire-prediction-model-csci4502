"""Ingest CLI. Runs locally or inside the CodeBuild project.

    python -m ingest all                    # everything, in order
    python -m ingest gridmet --only erc     # one gridMET variable
    python -m ingest landfire --only topo   # one LANDFIRE group
    python -m ingest integrate|catalog|viewer
    --force re-downloads / rebuilds objects that already exist.

Exit code 1 if any step failed (failures are listed; other steps still run).
"""
from __future__ import annotations

import argparse
import logging

from ingest import IngestResult
from ingest.catalog import register_catalog
from ingest.gridmet import ingest_gridmet
from ingest.integrate import integrate_all
from ingest.landfire import ingest_landfire
from ingest.mtbs import ingest_mtbs
from lakehouse.config import LANDFIRE_GROUPS
from lakehouse.viewer import publish_viewer

TARGETS = ("gridmet", "mtbs", "landfire", "integrate", "catalog", "viewer", "all")
ALL_ORDER = ("gridmet", "mtbs", "landfire", "integrate", "catalog", "viewer")


def _step(name: str, args):
    only = args.only
    if name == "gridmet":
        return ingest_gridmet(variables=[only] if only else None, force=args.force)
    if name == "mtbs":
        return ingest_mtbs(force=args.force)
    if name == "landfire":
        groups = {only: LANDFIRE_GROUPS[only]} if only else None
        return ingest_landfire(groups=groups, force=args.force)
    if name == "integrate":
        return integrate_all()
    if name == "catalog":
        return register_catalog()
    if name == "viewer":
        return publish_viewer()
    raise ValueError(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest", description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--only", help="one gridMET variable or one LANDFIRE group")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.only and args.target == "landfire" and args.only not in LANDFIRE_GROUPS:
        parser.error(f"unknown LANDFIRE group {args.only!r}; choose from {', '.join(LANDFIRE_GROUPS)}")

    steps = ALL_ORDER if args.target == "all" else (args.target,)
    total = IngestResult()
    for name in steps:
        try:
            out = _step(name, args)
        except Exception as err:  # one broken step must not hide the others' results
            logging.getLogger("ingest").exception("%s failed", name)
            total.failed.append((name, f"{type(err).__name__}: {err}"))
            continue
        if isinstance(out, IngestResult):
            total = total.merge(out)

    print(f"done: {len(total.records)} written, {len(total.skipped)} skipped, {len(total.failed)} failed")
    for what, why in total.failed:
        print(f"  FAILED {what}: {why}")
    return 1 if total.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
