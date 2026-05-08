2026-05-07 tick 28 (~23:55): T27 JIT build STATUS CHECK — still
in flight, no probe yet.

`bsn7ryi30` is alive (single icpx process active per MAX_JOBS=1).
Currently compiling `chunk_prefill_kernel_template_chunk_policy_head128_fttff.cpp`
(an `attn_kernels_xe_2` target).

**Progress estimate**:
- ninja reports 624 of 1308 total steps remain (`ninja -n`).
- 589 `.cpp.o` artifacts present in `build/temp` so far.
- We're past ~52% by step count; "remaining" matches what was
  built. At ~11 sec/step (MAX_JOBS=1 SYCL template pace), the
  ETA from now is ~1h54m → finish ~01:50 local.

`_xpu_C.abi3.so` mtime unchanged (still T26 volatile-temp build);
will only refresh at install step (~step 1300+). No probe possible
until then.

**Action**: reschedule wakeup at 3600s and check again. If still
running, reschedule again; if done, run
`vllm/.spec-gdn-load-readback.py` and decide T27 outcome.

Source state: pristine.

---

**Tick 29 update (~01:00 local)**: re-check after 60min wakeup.
Pace measured: 624 → 590 ninja steps in 60min at MAX_JOBS=1 = 34
steps/h. 542 of the 590 remaining are heavy `csrc/xpu/attn/`
chunk_prefill SYCL templates → ETA ~16 hours. Not viable.

**Critical insight**: `_xpu_C` (the gdn_attention extension) only
links `gdn_attn_kernels_xe_2` and `grouped_gemm_xe_2`. The 542
chunk_prefill files belong to `attn_kernels_xe_2` which is only
linked by `_vllm_fa2_C` (FA2 extension). Disabling
`FA2_KERNELS_ENABLED=OFF` skips both the FA2 extension target and
its 542 .cpp files. setup.py honors the env var
(`_is_enabled("FA2_KERNELS_ENABLED")`) and translates it to
`-DFA2_KERNELS_ENABLED=OFF` cmake arg.

Killed `bsn7ryi30` MAX_JOBS=1 build, restarted as `bdt43kc0q`
with `FA2_KERNELS_ENABLED=OFF MAX_JOBS=2 VLLM_XPU_AOT_DEVICES=
VLLM_XPU_XE2_AOT_DEVICES=`. CMake reconfigured: ninja remaining
count dropped from 590 → **44** (the small/cheap steps + link +
install). JIT mode re-verified. ETA ~10-15 min.
