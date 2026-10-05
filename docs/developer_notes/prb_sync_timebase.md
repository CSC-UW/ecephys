---
title: Probe sync table timebase
updated: 2026-09-24
---

# Probe sync table timebase

`prb_sync.{ap,lf}.htsv` (one row per file, `source` -> `target`, `slope`,
`intercept`) maps a non-reference probe's times onto the reference probe. Two
timebases exist for any SpikeGLX file:

- within-file seconds: `sample / imSampRate`, the origin of the `time` column in
  `*.barcodes.htsv` and of `rising` / `falling` in `*.ttls.htsv`
  (`ecephys.sync.load_sync_channel_from_sglx_imec`);
- experiment-acquisition seconds: within-file seconds plus the file's
  `expmtPrbAcqFirstTime` (`ecephys.wne.sglx.experiments`), how long that probe's
  acquisition had been running when the file began.

A per-file linear fit made on one timebase is wrong on the other by
`(slope - 1) * F_probe + (F_probe - F_ref)`, constant within a file and zero on
the reference probe.

## Convention

Generation has two stages (`generate_probe_sync_table`):

1. `do_session` fits each non-reference file onto the reference probe on
   within-file time, straight from `*.barcodes.htsv` / `*.ttls.htsv` in all
   three `imSyncType` branches, and writes `{session}.prb_sync.{stream}.htsv`
   to the project subject directory with `timebase = "withinFile"`. It depends
   only on the session's sync files, so every experiment containing the session
   reuses it. A file with too few events keeps NaN `slope` / `intercept`.
2. `do_experiment` converts each session table with `to_experiment_timebase`:
   the slope is unchanged and `intercept += F_target - slope * F_source`, with
   `F` from the experiment frame. NaN rows are then linearly interpolated
   between neighbours if the file is under 60 s (intercepts are near-constant
   on this timebase within an acquisition segment). The result is written as
   `prb_sync.{stream}.htsv` with `timebase = "expmtPrbAcq"`. `refit=False`
   reuses the session tables on disk instead of refitting.

The conversion is exact in the algebra and numerically: on Roland and Pier the
two-stage table matches a direct experiment-time fit to under 3e-9 s in
intercept and 2e-14 in slope.

`get_time2time`, `add_sample2time_columns`, `get_sample2time` and
`slice_table2times` apply `slope * t + intercept` to experiment-acquisition
inputs, unchanged. `generate_tdt_sync_table` fits on the same timebase.

## Guard

Every row carries `timebase = "expmtPrbAcq"` (`constants.PROBE_SYNC_TIMEBASE`).
`wne.sglx.utils.load_probe_sync_table` is the only reader
(`get_probe_sync_table` resolves the path from a project); it raises on a table
without the column or with any other value. Tables written before the column
existed were fitted on within-file time and must be regenerated before use.

## Un-finalized recordings

A file whose `.meta` lost `firstSample` has NaN `expmtPrbAcqFirstTime`. Its
within-file session row is still fitted, but no experiment-timebase row can be
made from it: `to_experiment_timebase` raises `UnfinalizedRecordingError`
through `require_acq_time`, and `get_time2time` raises for a probe with any such
file. Repair the meta
(`ecephys.sglx.repair_metadata`) or exclude the session.

## Checking a table

Barcodes shared by two probes are one event. For each non-reference file, map
`time + F_probe` through `get_time2time` and compare with the reference file's
`time + F_ref`; the per-file median should be at the fit residual (tens of
microseconds). `tests/test_prb_sync_timebase.py` does this on production data
(`requires_nfs`).
