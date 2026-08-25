# minos_convert

Batch-converts a tree of MINOS SNTP ROOT files into a mirrored tree of HDF5,
for long-term storage that can be read without ROOT.

```bash
python convert_sntp_to_hdf5.py /data/sntp /archive/hdf5
```

Input structure is preserved:

```
/data/sntp/2010/run1/f21….sntp.dogwood5.0.root
  ->  /archive/hdf5/2010/run1/f21….sntp.dogwood5.0.h5
```

Only the final `.root` is replaced. Stripping the whole `.sntp.dogwood5.0`
tail would read better, but two inputs differing only in those middle
components would then collide on one output name.

## Requirements

Python 3.10 or 3.11, and a checkout of
[oscana](https://github.com/aditya-marathe/oscana) — it does the actual
reading and writing. It is not on PyPI, so the script adds its `src/`
directory to `sys.path`; by default it looks for a sibling checkout at
`../oscana/src`, override with `--oscana-src`.

Any environment with oscana's dependencies installed works, including
oscana's own `.venv`:

```bash
/path/to/oscana/.venv/bin/python convert_sntp_to_hdf5.py in/ out/
```

## What gets written

By default the curated "keep" set for long-term storage — hit-level data
plus MC truth:

| Collection | Contents |
|------------|----------|
| `IMAGE_ALL_VARIABLES` | digitised strip hits: plane, strip, view, z, pulse height, timing |
| `MC_TRUTH_EVENT_VARIABLES` | per-event interaction truth (x, y, Q², W², channel, cross sections) |
| `MC_INTERACTION_VARIABLES` | interaction type and neutrino flavour |
| `MC_4MOMENTUM_VARIABLES` | truth 4-vectors |
| `MC_PARTICLE_VARIABLES` | the `stdhep` truth particle stack |

Override with `--variables` (comma-separated collection names from
`oscana.constants`). See `src/oscana/README.md` in the oscana checkout for
what each variable means.

Columns that are exact functions of another are left out and rebuilt on
load — currently `stp.z` and `stp.planeview`, both derivable from
`stp.plane`. Pass `--no-drop-derived` to write them literally. Either way
the data you read back is identical; oscana verifies the mapping on the
data being written and keeps the column if it does not hold, so files whose
geometry differs are never silently corrupted.

Reading the result back:

```python
import oscana, pandas as pd

dh = oscana.data.DataHandler[pd.DataFrame](
    variables=oscana.IMAGE_ALL_VARIABLES.uproot
)
dh.io.from_hdf5(files=["/archive/hdf5/2010/run1/f21….h5"])
```

## Options

| Flag | |
|------|--|
| `--pattern GLOB` | which files to pick up (default `**/*.root`) |
| `--variables NAMES` | comma-separated oscana collections |
| `--compression {gzip,lzf,none}` | default `gzip` |
| `--no-drop-derived` | store derived columns literally |
| `--max-events N` | keep only the first N events per file (testing) |
| `--overwrite` | reconvert files whose output already exists |
| `--verify` | reload each output and check the event count |
| `--dry-run` | list planned work and stop |
| `--oscana-src PATH` | oscana's `src/` (default `../oscana/src`) |

## Behaviour worth knowing

**One process per file.** A single file needs roughly 1.2 GB resident, and
Python does not reliably hand that back between iterations, so a long
in-process loop climbs until it is killed. Each conversion therefore runs in
its own subprocess. It costs a couple of seconds of startup per file against
a conversion measured in minutes, and keeps memory flat across a run of any
length.

**Resumable.** Files whose output already exists are skipped unless
`--overwrite`, so an interrupted run can simply be repeated.

**One bad file does not stop the batch.** Failures are collected and printed
at the end with the tail of their error, and the exit status is non-zero, so
it composes in a pipeline.

**Real data vs simulation.** The default set includes `mc.*`. Real-data
files carry those branches zero-filled, so they convert without complaint.
If a branch is genuinely missing the file is reported as failed, naming the
variable, rather than guessing — narrow `--variables` in that case.

## Related

- [oscana](https://github.com/aditya-marathe/oscana) — the library doing the
  I/O; `src/oscana/README.md` documents every variable and which are
  redundant.
- `minos_data_storage` — earlier exploration of the same problem in Parquet,
  including `SCHEMA.md` (what each branch means, verified) and
  `validate_redundancy.py` (checks the redundancy claims against a file).
