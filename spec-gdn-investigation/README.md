# spec-gdn investigation

Historical record of the loop that fixed the SYCL `gdn_attention` spec-decode
bug. Lives in the meta-repo because these are agent breadcrumbs, not vllm
source.

## Layout

- `progress/` — tick-by-tick journals from the loop run. Read
  [`progress/README.md`](progress/README.md) for the closing summary; tick
  files are bottom-up chronological (read upward for context).
- `probes/` — investigation scripts that captured GPU state, diffed kernel
  output against the FLA Triton oracle, ran ISA disassembly, etc. Originally
  lived as dotfiles (`.spec-gdn-*.py`) at the vllm fork root; renamed here
  to drop the leading dot since they're in their own dedicated dir.

## Key probes

- `probes/spec-gdn-coreattn-diff.py` — per-slot conv + ssm cell-by-cell diff
  vs FLA. The trustworthy primitive — capture real prod inputs, replay
  against the slow reference, diff. Used in ticks 36-39 to find the layout
  bug.
- `probes/spec-gdn-capture.sh` — wraps `VLLM_XPU_DUMP_SPEC_GDN=<dir> vllm
  serve …` to dump real (input, output) tuples for offline replay.
- `probes/spec-gdn-load-readback.py` — **WARNING: this probe lied for 17
  ticks.** It set inputs to `-100` to force `sigmoid(-100) ≈ 0`. On the
  Intel Xe2 GPU under fast-math, `sigmoid(-100)` returns small-but-nonzero,
  so the "zeroed" term wasn't zero, generating drift that looked like a
  kernel bug. Don't trust readings from synthetic-zeroing probes without
  verifying their numerical assumption holds on device. See
  `progress/tick-34.md`.

## What the loop accomplished

- Fixed a spec-decode-aware GDN attention layout bug (FLA expects
  `cache_indices[batch, 0]` to hold the rolled history; SYCL was scattering
  per-token snapshots across K slots). Two-pass kernel: stage candidates,
  consolidate to slot 0.
- Rebuilt the in-vllm side as `xpu: spec-decode-aware GDN attention
  dispatcher + replay test` (in `../vllm`).
- Rebuilt the kernel side as `xpu: spec-decoding-aware GDN attention
  kernel` (in `../vllm-xpu-kernels`).

Both committed clean and atomic on each fork's `main`.
