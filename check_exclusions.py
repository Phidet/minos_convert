#!/usr/bin/env python3
"""Verify, per file, the assumptions behind every branch we leave out.

The archive drops some branches because of a claim about their *contents* --
that they are empty, or a constant sentinel, or an exact copy of something
we do keep.

Every such claim is declared in ASSUMPTIONS below. A claim that fails stops the file being
converted. This does not test branches that are dropped as a policy choice (e.g. the old reconstruction branches).

python check_exclusions.py FILE.sntp.root [--check-events N]
"""

from __future__ import annotations

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

TREE = "NtpSt"
MC = "NtpStRecord/mc/mc."
STP = "NtpStRecord/stp/stp."
PARTICLE = "NtpStRecord/stdhep/stdhep."

# Float comparisons use a tolerance: the values are expected to be identical
# copies, but float32 -> float64 promotion during reads can move the last
# bit, which says nothing about whether the data matches.
TOL = 1e-4


class Failure(Exception):
    """An assumption about an excluded branch did not hold."""


def _first(tree, name, n):
    """A field of the length-0-or-1 `mc` collection, one value per event."""
    return ak.firsts(tree[MC + name].array(entry_stop=n), axis=1)


# --------------------------------------------------------------------------
# Checks. Each returns a short message on success, or raises Failure.
# --------------------------------------------------------------------------


def check_empty(tree, n, branch: str, label: str) -> str:
    """The branch exists but holds no entries in any event."""
    leaves = [k for k in tree.keys(recursive=True) if k.startswith(branch)]
    if not leaves:
        return f"{label}: branch absent"

    counts = ak.to_numpy(
        ak.num(tree[sorted(leaves, key=len)[0]].array(entry_stop=n), axis=1)
    )
    if counts.max() != 0:
        raise Failure(
            f"{label} is NOT empty in this file: up to {counts.max()} entries "
            f"per event, {np.mean(counts > 0):.1%} of events populated. It is "
            "excluded on the assumption that it is always empty, so this file "
            "would lose real data."
        )
    return f"{label}: empty in all {len(counts)} events"


def check_constant(tree, n, field: str, label: str) -> str:
    """The field never varies -- it carries no information."""
    values = ak.to_numpy(_first(tree, field, n), allow_missing=True).astype(float)
    finite = values[~np.isnan(values)]
    distinct = np.unique(finite)
    if len(distinct) > 1:
        raise Failure(
            f"{label} takes {len(distinct)} distinct values here "
            f"(e.g. {distinct[:5].tolist()}). It is excluded on the assumption "
            "that it is a single unset sentinel, so this file would lose "
            "real information."
        )
    return f"{label}: constant ({distinct[0] if len(distinct) else 'n/a'})"


def check_function_of(tree, n, derived: str, source: str, label: str) -> str:
    """`derived` is fixed by `source` -- no source value maps to two values."""
    src = ak.to_numpy(ak.flatten(tree[STP + source].array(entry_stop=n), axis=1))
    drv = ak.to_numpy(ak.flatten(tree[STP + derived].array(entry_stop=n), axis=1))

    order = np.argsort(src, kind="stable")
    src_s, drv_s = src[order], drv[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(src_s)) + 1))
    groups = np.split(drv_s, starts[1:])

    bad = [
        int(src_s[starts[i]])
        for i, g in enumerate(groups)
        if len(np.unique(g)) != 1
    ]
    if bad:
        raise Failure(
            f"{label}: {len(bad)} value(s) of {source} map to more than one "
            f"{derived} (e.g. {bad[:5]}). It is excluded on the assumption "
            f"that {source} fixes it, so this file would lose real "
            "information."
        )
    return f"{label}: fixed by {source} ({len(groups)} distinct values)"


def _matched_row(pdg, status, target_pdg, target_status, values):
    """Values from the first particle row matching (pdg, status)."""
    local = ak.local_index(pdg, axis=1)
    mask = (pdg == target_pdg) & (status == target_status)
    idx = ak.firsts(local[mask], axis=1)
    found = ~ak.is_none(idx)
    picked = ak.firsts(values[idx[:, None] == local], axis=1)
    return found, picked


def _compare(label, stored, predicted, found, what):
    stored_np = ak.to_numpy(stored)
    nonzero = np.any(stored_np != 0, axis=1) if stored_np.ndim > 1 else stored_np != 0
    if not nonzero.any():
        return f"{label}: no non-zero example to check"

    missing = nonzero & ~ak.to_numpy(found)
    if missing.any():
        raise Failure(
            f"{label}: {int(missing.sum())} event(s) have a value but no "
            f"matching {what}. It is excluded on the assumption that it "
            "duplicates that row, so this file would lose real information."
        )

    pred = ak.to_numpy(ak.fill_none(predicted, [np.nan] * 4)) \
        if not isinstance(predicted, np.ndarray) else predicted
    diff = np.abs(stored_np[nonzero] - pred[nonzero])
    if diff.size and np.nanmax(diff) > TOL:
        n_bad = int(np.sum(np.any(diff > TOL, axis=1)))
        raise Failure(
            f"{label}: {n_bad}/{int(nonzero.sum())} event(s) differ from "
            f"{what} by more than {TOL} (max {np.nanmax(diff):.6g}). The "
            "duplication assumption does not hold for this file."
        )
    return f"{label}: matches {what} on {int(nonzero.sum())} events"


def check_p4neu(tree, n, label: str) -> str:
    pdg = tree[PARTICLE + "IdHEP"].array(entry_stop=n)
    status = tree[PARTICLE + "IstHEP"].array(entry_stop=n)
    p4 = tree[PARTICLE + "p4[4]"].array(entry_stop=n)
    found, picked = _matched_row(pdg, status, _first(tree, "inu", n), 0, p4)
    return _compare(label, _first(tree, "p4neu[4]", n), picked, found,
                    "the stdhep neutrino row")


def check_vtx(tree, n, label: str) -> str:
    pdg = tree[PARTICLE + "IdHEP"].array(entry_stop=n)
    status = tree[PARTICLE + "IstHEP"].array(entry_stop=n)
    vtx = tree[PARTICLE + "vtx[4]"].array(entry_stop=n)
    found, picked = _matched_row(pdg, status, _first(tree, "inu", n), 0, vtx)

    stored = np.stack(
        [ak.to_numpy(_first(tree, f"vtx{c}", n)) for c in "xyz"], axis=1
    )
    pred = ak.to_numpy(ak.fill_none(picked, [np.nan] * 4))[:, :3]
    found_np = ak.to_numpy(found)
    if not found_np.all():
        raise Failure(
            f"{label}: {int((~found_np).sum())} event(s) have no stdhep "
            "neutrino row at all."
        )
    diff = np.abs(stored - pred)
    if np.nanmax(diff) > TOL:
        raise Failure(
            f"{label}: differs from the stdhep neutrino row's vtx by up to "
            f"{np.nanmax(diff):.6g}."
        )
    return f"{label}: matches the stdhep neutrino row on {len(stored)} events"


def check_a_z(tree, n, label: str) -> str:
    pdg = tree[PARTICLE + "IdHEP"].array(entry_stop=n)
    status = tree[PARTICLE + "IstHEP"].array(entry_stop=n)
    is_nucleus = (pdg > 1e9) & (status == 0)
    nucleus_pdg = ak.to_numpy(
        ak.fill_none(ak.firsts(pdg[is_nucleus], axis=1), 0)
    )
    found = nucleus_pdg != 0

    # No nucleus row means a free proton: hydrogen, the verified fallback.
    a = np.where(found, (nucleus_pdg // 1_000_000) % 1000, 1)
    z = np.where(found, (nucleus_pdg // 1_000) % 1000, 1)

    stored_a = ak.to_numpy(_first(tree, "a", n))
    stored_z = ak.to_numpy(_first(tree, "z", n))
    bad = (stored_a != a) | (stored_z != z)
    if bad.any():
        raise Failure(
            f"{label}: {int(bad.sum())} event(s) do not match the nucleus-row "
            "decode (or the hydrogen fallback). The duplication assumption "
            "does not hold for this file."
        )
    return (f"{label}: decodes from the stdhep nucleus row on {len(stored_a)} "
            f"events ({int((~found).sum())} via the hydrogen fallback)")


def check_lepton_p4(tree, n, field: str, lepton_pdg: int, label: str) -> str:
    """Match by momentum, then sign the energy by the matched particle's own
    charge. NOT "first matching row in stack order" -- that looked right on a
    hand-picked sample and is wrong in general."""
    pdg = tree[PARTICLE + "IdHEP"].array(entry_stop=n)
    status = tree[PARTICLE + "IstHEP"].array(entry_stop=n)
    p4 = tree[PARTICLE + "p4[4]"].array(entry_stop=n)

    is_lepton = (abs(pdg) == lepton_pdg) & (status == 1)
    cand_p4, cand_pdg = p4[is_lepton], pdg[is_lepton]

    stored = _first(tree, field, n)
    target = stored[:, :3]
    match = ak.all(
        abs(cand_p4[:, :, :3] - target[:, np.newaxis, :]) < TOL, axis=2
    )
    matched_pdg = ak.firsts(cand_pdg[match], axis=1)
    matched_p4 = ak.firsts(cand_p4[match], axis=1)

    pred = ak.to_numpy(ak.fill_none(matched_p4, [np.nan] * 4)).copy()
    sign = np.where(
        ak.to_numpy(ak.fill_none(matched_pdg, 0)) == lepton_pdg, -1, 1
    )
    pred[:, 3] *= sign
    return _compare(label, stored, pred, ~ak.is_none(matched_pdg),
                    f"an stdhep |pdg|=={lepton_pdg} row")


# --------------------------------------------------------------------------
# The assumptions themselves. One line per excluded branch.
# --------------------------------------------------------------------------

ASSUMPTIONS = [
    ("digihit",     lambda t, n: check_empty(t, n, "NtpStRecord/digihit/digihit.", "digihit")),
    ("deadchips",   lambda t, n: check_empty(t, n, "NtpStRecord/deadchips/deadchips.", "deadchips")),
    ("mc.iboson",   lambda t, n: check_constant(t, n, "iboson", "mc.iboson")),
    ("stp.z",       lambda t, n: check_function_of(t, n, "z", "plane", "stp.z")),
    ("mc.p4neu",    lambda t, n: check_p4neu(t, n, "mc.p4neu[4]")),
    ("mc.vtx",      lambda t, n: check_vtx(t, n, "mc.vtxx/y/z")),
    ("mc.a, mc.z",  lambda t, n: check_a_z(t, n, "mc.a, mc.z")),
    ("mc.p4mu2",    lambda t, n: check_lepton_p4(t, n, "p4mu2[4]", 13, "mc.p4mu2[4]")),
    ("mc.p4el1",    lambda t, n: check_lepton_p4(t, n, "p4el1[4]", 11, "mc.p4el1[4]")),
    ("mc.p4el2",    lambda t, n: check_lepton_p4(t, n, "p4el2[4]", 11, "mc.p4el2[4]")),
    ("mc.p4tau",    lambda t, n: check_lepton_p4(t, n, "p4tau[4]", 15, "mc.p4tau[4]")),
]


def check_file(path, check_events: int | None = 5000, verbose: bool = False):
    """Run every assumption. Returns a list of failure messages (empty if OK)."""
    failures = []

    # Opening can fail on its own -- a truncated or non-ROOT file. Report
    # that like any other failed assumption rather than letting it escape
    # and take the whole batch down with it.
    try:
        handle = uproot.open(path)
    except Exception as exc:
        return [f"could not open as a ROOT file -- {exc!r}"]

    with handle as f:
        try:
            tree = f[TREE]
        except Exception as exc:
            return [f"no {TREE!r} tree -- {exc!r}"]

        n = min(check_events, tree.num_entries) if check_events else None
        for name, check in ASSUMPTIONS:
            try:
                message = check(tree, n)
                if verbose:
                    print(f"    ok   {message}")
            except Failure as exc:
                failures.append(str(exc))
            except Exception as exc:  # a check that cannot run is not a pass
                failures.append(f"{name}: check could not run -- {exc!r}")
    return failures


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    n = 5000
    if "--check-events" in argv:
        i = argv.index("--check-events")
        n = int(argv[i + 1])
        del argv[i:i + 2]
    if not argv:
        print(__doc__)
        return 1

    status = 0
    for path in argv:
        print(f"\n=== {path} ===")
        failures = check_file(Path(path), check_events=n, verbose=True)
        if failures:
            status = 1
            print(f"  {len(failures)} ASSUMPTION(S) BROKEN:")
            for message in failures:
                print(f"    - {message}")
        else:
            print("  all assumptions hold")
    return status


if __name__ == "__main__":
    sys.exit(main())
