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
| `IMAGE_ALL_VARIABLES` | digitised strip hits: plane, strip, view, pulse height, timing |
| `MC_TRUTH_EVENT_VARIABLES` | per-event interaction truth (x, y, Q², W², channel, cross sections) |
| `MC_INTERACTION_VARIABLES` | interaction type and neutrino flavour |
| `MC_4MOMENTUM_VARIABLES` | truth 4-vectors |
| `MC_PARTICLE_VARIABLES` | the `stdhep` truth particle stack |
| `MC_STRIP_TRUTH_VARIABLES` | per-hit truth: which particles lit each strip, and by how much |
| `DETECTOR_STATE_VARIABLES` | magnet coil current, MIP→GeV calibration constant, HV status |
| `DAQ_CONTEXT_VARIABLES` | beam spill, trigger and absolute timing (real data only; unset in MC) |
| `VETO_SHIELD_VARIABLES` | raw veto shield hits |

Override with `--variables` (comma-separated collection names from
`oscana.constants`). See `src/oscana/README.md` in the oscana checkout for
what each variable means.

Everything requested is written out as-is. Some of it is redundant in
principle — `stp.planeview` is a pure function of `stp.plane`, for
instance — but oscana stores it rather than reconstructing it on load, and
the README there records which fields those are and how they relate.

Reconstruction output is deliberately absent: no tracks, showers, slices or
clusters, and not the reconstructed vertex. The aim is to preserve what the
experiment and the simulation recorded, so a future analysis can start from
the data rather than inherit MINOS's own.

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
| `--max-events N` | keep only the first N events per file (testing) |
| `--overwrite` | reconvert files whose output already exists |
| `--verify` | reload each output and check the event count |
| `--dry-run` | list planned work and stop |
| `--no-check-exclusions` | skip the checks described below |
| `--check-events N` | events to sample for those checks (default 5000) |
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

**Excluded branches are checked, not assumed.** Some branches are dropped
because of a claim about their contents — that they are empty, a constant
sentinel, or an exact copy of something kept. Before converting, each file
is tested against those claims (`check_exclusions.py`), and refused if any
fails. Without that, a file where `digihit` was actually populated, or where
`mc.p4neu` did not match the truth particle table, would be silently
stripped of real data. The reconstruction chain is not checked: it is
dropped by policy, so there is no claim to test.

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
