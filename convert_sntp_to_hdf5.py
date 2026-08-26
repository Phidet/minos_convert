#!/usr/bin/env python3
"""Convert a tree of MINOS SNTP ROOT files into a mirrored tree of HDF5.

Walks INPUT_DIR for ROOT files, converts each through oscana's HDF5 writer,
and writes the result to the same relative location under OUTPUT_DIR.

    convert_sntp_to_hdf5.py /data/sntp /archive/hdf5

Each file is converted in a separate process. That is deliberate: one file
needs roughly 1.2 GB resident, and Python does not reliably return that to
the operating system between iterations, so a long in-process loop creeps
upwards until it is killed. A subprocess always gives the memory back. The
cost is a couple of seconds of interpreter startup per file, against a
conversion measured in minutes.

Interrupted runs resume: a file whose output already exists is skipped
unless --overwrite. A file that fails does not stop the batch; failures are
collected and reported at the end, and the exit status is non-zero.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from check_exclusions import check_file

# Collections from oscana.constants making up the "keep" set for long-term
# storage. See README.md in the oscana package for what each variable means
# and which are redundant.
DEFAULT_VARIABLES = [
    "IMAGE_ALL_VARIABLES",
    "MC_TRUTH_EVENT_VARIABLES",
    "MC_INTERACTION_VARIABLES",
    "MC_4MOMENTUM_VARIABLES",
    "MC_PARTICLE_VARIABLES",
    "MC_STRIP_TRUTH_VARIABLES",
    "DETECTOR_STATE_VARIABLES",
    "DAQ_CONTEXT_VARIABLES",
    "VETO_SHIELD_VARIABLES",
    "IMAGE_RAW_VARIABLES",
    "MC_PARTICLE_LINEAGE_VARIABLES",
    "MC_FLUX_VARIABLES",
]

# Runs in a fresh interpreter, one per input file.
#
# Two things here are not obvious, and both come from how oscana expects to
# be set up rather than from anything we want:
#
#   * `from_sntp` resolves its argument through the environment, not the
#     filesystem -- `_get_dir_from_env` does `os.environ.get(name)` and
#     errors if the name is absent. There is no path fallback, so we put the
#     path into the environment under a synthetic key and hand over the key.
#
#   * `oscana.init()` refuses to start without a .env file it can find and a
#     logs directory that already exists, and python-dotenv only searches
#     from the working directory when it believes it is interactive (hence
#     setting sys.ps1). So the worker builds a scratch directory holding
#     both and works from there. Output still goes to the real destination.
WORKER = r'''
import os, sys, tempfile, json
from pathlib import Path

sys.ps1 = ">>> "                       # make python-dotenv search from cwd
sys.path.insert(0, {oscana_src!r})

src, dst = Path({src!r}), Path({dst!r})
key = "SNTP_INPUT_FILE"

scratch = Path(tempfile.mkdtemp(prefix="minos_convert_"))
(scratch / ".env").write_text(f"{{key}}={{src}}\n")
(scratch / "logs").mkdir()
os.environ[key] = str(src)
os.chdir(scratch)

import pandas as pd
import oscana

oscana.init(logs_dir="./logs/")

variables = []
for name in {variables!r}:
    variables += getattr(oscana, name).uproot

dh = oscana.data.DataHandler[pd.DataFrame](variables=variables)
dh.io.from_sntp(files=[key])

max_events = {max_events!r}
if max_events is not None:
    import logging, warnings
    root = logging.getLogger("Root")
    level = root.level
    root.setLevel(logging.ERROR)          # unlock_data warns by design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dh.unlock_data()
        dh.data = dh.data.head(max_events)
        dh.lock_data()
    root.setLevel(level)

n_events = len(dh.data)
n_columns = len(dh.data.columns)

dst.parent.mkdir(parents=True, exist_ok=True)
dh.io.to_hdf5(file=dst, compression={compression!r})

verified = None
if {verify!r}:
    check = oscana.data.DataHandler[pd.DataFrame](variables=variables)
    check.io.from_hdf5(files=[str(dst)])
    verified = len(check.data)

print("RESULT " + json.dumps(
    {{"events": n_events, "columns": n_columns, "verified": verified}}
))
'''


def human(n_bytes: float) -> str:
    """Format a byte count for a progress line."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} PB"


def output_path_for(src: Path, input_dir: Path, output_dir: Path) -> Path:
    """Mirror `src` under `output_dir`, swapping the final .root for .h5.

    Only the last suffix is replaced. Stripping the whole `.sntp.dogwood5.0`
    tail would read more nicely, but two inputs differing only in those
    middle components would then collide on one output name.
    """
    relative = src.relative_to(input_dir)
    return output_dir / relative.with_suffix(".h5")


def convert_one(src: Path, dst: Path, args: argparse.Namespace) -> dict:
    """Convert a single file in a subprocess. Never raises for a bad file."""
    code = WORKER.format(
        oscana_src=str(args.oscana_src),
        src=str(src),
        dst=str(dst),
        variables=args.variables,
        compression=None if args.compression == "none" else args.compression,
        max_events=args.max_events,
        verify=args.verify,
    )

    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0 or not dst.exists():
        # Keep the tail only: oscana logs a lot before it fails.
        detail = (completed.stderr or "").strip().splitlines()
        return {
            "ok": False,
            "elapsed": elapsed,
            "error": detail[-1] if detail else
                     f"exited {completed.returncode} without writing output",
            "stderr": "\n".join(detail[-15:]),
        }

    payload = {}
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            import json

            payload = json.loads(line[len("RESULT "):])

    return {"ok": True, "elapsed": elapsed, **payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="tree of ROOT files")
    parser.add_argument("output_dir", type=Path, help="where to mirror it")
    parser.add_argument(
        "--pattern", default="**/*.root",
        help="glob relative to INPUT_DIR (default: %(default)s)",
    )
    parser.add_argument(
        "--variables", default=",".join(DEFAULT_VARIABLES),
        help="comma-separated oscana collection names "
             "(default: the curated storage set)",
    )
    parser.add_argument(
        "--compression", default="gzip", choices=("gzip", "lzf", "none"),
        help="HDF5 compression (default: %(default)s)",
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="keep only the first N events per file (for testing)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="reconvert files whose output already exists",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="reload each output and check the event count matches",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list what would be done, then stop",
    )
    parser.add_argument(
        "--no-check-exclusions", action="store_true",
        help="skip the checks that the dropped branches really are empty, "
             "constant, or duplicated elsewhere",
    )
    parser.add_argument(
        "--check-events", type=int, default=5000,
        help="events to sample for those checks (default: %(default)s)",
    )
    parser.add_argument(
        "--oscana-src", type=Path,
        default=Path(__file__).resolve().parent.parent / "oscana" / "src",
        help="path to oscana's src/ (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    args.variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.oscana_src = args.oscana_src.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"error: {args.input_dir} is not a directory", file=sys.stderr)
        return 2

    if not (args.oscana_src / "oscana").is_dir():
        print(
            f"error: no oscana package under {args.oscana_src}\n"
            "       point --oscana-src at the src/ directory of an oscana "
            "checkout",
            file=sys.stderr,
        )
        return 2

    sources = sorted(args.input_dir.glob(args.pattern))
    if not sources:
        print(f"No files matching {args.pattern!r} under {args.input_dir}")
        return 0

    print(f"{len(sources)} file(s) under {args.input_dir}")
    print(f"variables: {', '.join(args.variables)}")
    print(f"compression: {args.compression}\n")

    converted, skipped, failures = [], [], []
    bytes_in = bytes_out = 0

    for i, src in enumerate(sources, start=1):
        dst = output_path_for(src, args.input_dir, args.output_dir)
        label = f"[{i}/{len(sources)}] {src.relative_to(args.input_dir)}"

        if dst.exists() and not args.overwrite:
            print(f"{label}\n    skipped, output exists ({dst})")
            skipped.append(src)
            continue

        if args.dry_run:
            print(f"{label}\n    would write {dst}")
            continue

        print(f"{label}", flush=True)

        if not args.no_check_exclusions:
            broken = check_file(src, check_events=args.check_events)
            if broken:
                print(f"    REFUSED: {len(broken)} assumption(s) about "
                      "excluded branches do not hold for this file")
                for message in broken:
                    print(f"      - {message}")
                failures.append((src, {"error": "exclusion checks failed",
                                       "stderr": "\n".join(broken)}))
                continue

        result = convert_one(src, dst, args)

        if not result["ok"]:
            print(f"    FAILED after {result['elapsed']:.1f}s: "
                  f"{result['error']}")
            failures.append((src, result))
            continue

        in_size, out_size = src.stat().st_size, dst.stat().st_size
        bytes_in += in_size
        bytes_out += out_size

        note = ""
        if result.get("verified") is not None:
            match = result["verified"] == result["events"]
            note = (f", verified {result['verified']:,} events"
                    if match else
                    f", VERIFY MISMATCH {result['verified']:,} != "
                    f"{result['events']:,}")
            if not match:
                failures.append((src, {"error": "verify mismatch"}))

        print(f"    {result['events']:,} events, {result['columns']} columns"
              f"{note}")
        print(f"    {human(in_size)} -> {human(out_size)} "
              f"({out_size / in_size:.1%}) in {result['elapsed']:.1f}s")
        converted.append(src)

    if args.dry_run:
        print(f"\nDry run: {len(sources) - len(skipped)} to convert, "
              f"{len(skipped)} already present.")
        return 0

    print(f"\n{'-' * 60}")
    print(f"converted {len(converted)}, skipped {len(skipped)}, "
          f"failed {len(failures)}")
    if bytes_in:
        print(f"total {human(bytes_in)} -> {human(bytes_out)} "
              f"({bytes_out / bytes_in:.1%})")

    if failures:
        print("\nFailures:")
        for src, result in failures:
            print(f"  {src}")
            print(f"    {result.get('error', 'unknown')}")
            if result.get("stderr"):
                for line in result["stderr"].splitlines()[-5:]:
                    print(f"      | {line}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
