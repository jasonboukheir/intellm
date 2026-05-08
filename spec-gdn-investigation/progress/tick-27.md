2026-05-07 tick 27: T27 build setup — wrapper bug diagnosed, real
JIT-only build now in flight.

**The wrapper bug**: `vllm-xpu-build` (Nix store wrapper script)
contains:
```sh
export VLLM_XPU_AOT_DEVICES="${VLLM_XPU_AOT_DEVICES:-bmg}"
export VLLM_XPU_XE2_AOT_DEVICES="${VLLM_XPU_XE2_AOT_DEVICES:-bmg}"
```
The `${VAR:-default}` syntax substitutes default when VAR is unset
**OR EMPTY**. So `VLLM_XPU_AOT_DEVICES="" vllm-xpu-build` rewrites
empty → `bmg`, defeating the JIT-only intent. The kicked-off
"T27 build" from end of tick 26 (background task `b0l2pa007`) ran
~60 minutes as **AOT**, confirmed by inspecting live `icpx`
command line: `-fsycl-targets=spir64_gen` was present.

**Recovery**: killed b0l2pa007 + all live `icpx`/`ninja`/`pip
install` host- and container-side; restarted via direct `docker
exec` (bypasses the wrapper):
```
docker exec --workdir /workspace/vllm-xpu-kernels \
  -e VLLM_XPU_AOT_DEVICES= \
  -e VLLM_XPU_XE2_AOT_DEVICES= \
  -e MAX_JOBS=2 \
  -e CMAKE_BUILD_TYPE=Debug \
  vllm-dev \
  sh -c 'rm -rf build && pip install -e . --no-build-isolation'
```
`docker exec -e VAR=` sets the variable to empty *inside* the
container, which is what cmake needs. CMakeLists.txt:184-200 does
`if(DEFINED ENV{VLLM_XPU_AOT_DEVICES}) set(AOT_DEVICES ...)` then
`if(AOT_DEVICES)` — the second test is false on empty string, so
`-fsycl-targets=spir64_gen` is dropped.

**Verification (post-restart, monitor)**: live `icpx` command line
contains `-fsycl` but NOT `-fsycl-targets=spir64_gen`. JIT mode
confirmed.

**T27 build attempts**:
1. First attempt `b80v2uk7n` (MAX_JOBS=2, JIT-only, started
   ~22:38). **OOM-killed at step 659/1308** on
   `grouped_gemm_xe2.cpp`: `icpx: error: unable to execute
   command: Killed` after the clang frontend got SIGKILL. Host
   has 93 Gi total / 0 swap; two parallel SYCL-template
   compiles spiked past available memory. Build dir survived
   with JIT cmake config intact.
2. Resumed as `bsn7ryi30` at **MAX_JOBS=1** (~23:21 local) —
   keeps the build dir, picks up incrementally from ~50% step
   count. JIT mode re-verified via live icpx inspection. Single-
   job build is slower per-step but memory-safe.

**Next tick (28)**:
1. `tail` the build log + check `_xpu_C.abi3.so` mtime to confirm
   completion.
2. Run `vllm/.spec-gdn-load-readback.py` against the layer-0 worst
   capture under g=1, beta=0.
3. Decide:
   - drift ≈ 0 → bug is purely IGC-AOT. Move toward shipping
     JIT-only for `gated_delta_rule_kernel` on Xe2 + filing Intel
     bug. Rung 4 likely turns GREEN with the workaround.
   - drift ≈ 9.22 (or ≈ 8.95 like volatile-temp) → IGC backend is
     the same regardless of mode. Pivot to source-level RA
     decouple ladder (struct-wrap spec params, then SIMD-16, then
     IGC env workarounds).

**Source state**: pristine.

**Wrapper feedback** (consider saving as a build-cache-feedback
update): always pass empty AOT vars via `docker exec -e VAR=`
directly when bypassing AOT is required; the `vllm-xpu-build`
wrapper's `:-bmg` fallback silently re-enables AOT.
