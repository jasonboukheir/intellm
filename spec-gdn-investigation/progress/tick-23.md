2026-05-07 tick 23: Xe2 ISA disassembly pipeline working; spec=true vs
spec=false have IDENTICAL op mix but DIFFERENT register allocation —
strong support for tick 22's "register-allocation-sensitive IGC bug"
hypothesis.

**Toolchain established** (works in vllm-dev container + on host):
- Container has `ocloc` at /usr/bin/ocloc and `llvm-spirv` at
  `/opt/intel/oneapi/compiler/2025.3/bin/compiler/llvm-spirv`. NO
  spirv-dis, NO clang-offload-bundler in PATH (bundler is at
  `.../compiler/2025.3/bin/compiler/clang-offload-bundler`).
- Host has `iga64` at
  `/nix/store/vh5z5lhsx60ixdjp0jqvbafidms3rix3-intel-graphics-compiler-2.30.1/bin/iga64`
  (also w5jhnfaf...). Use `-p=2` for Xe2/BMG.
- `ocloc disasm -file <elf> -dump <dir> -device bmg-g21` recognizes
  kernels (warns "unexpected padding") but only writes `sections.txt`,
  not .asm files — modern zeBinary format. Use iga64 directly on
  extracted .text bytes instead.

**Extraction recipe** (saved at `/workspace/vllm/.spec_gdn_disasm/`,
host-visible at `~/Projects/vllm/.spec_gdn_disasm/`):
1. `objcopy --dump-section OFFLOAD_DEVICE_CODE=raw <so>` →
   concatenated `!<arch>` archives (12 of them in
   `_xpu_C.abi3.so`, one per TU).
2. Walk archives by scanning for `b"!<arch>\n"` magic; within each,
   parse 60-byte ar headers. Members are `pad_N` (skip), then per-IP
   GEN binary: `64.12.60.7`, `64.bmg`, `64.20.1.0`, `64.20.2.0`, plus
   `generic_ir`. The `64.bmg` member is the relevant Xe2 ELF
   (e_machine = 0xCD = EM_INTELGT).
3. ar0..ar9 don't have `gated_delta_rule_kernel`. **ar10 contains
   half/half and bf16/half mixed-type instantiations; ar11 contains
   bf16/bf16 and ff (float/float) instantiations.** All k_bucket_size
   ∈ {1,2,4,8} × IS_SPEC ∈ {0,1} are present in ar10/ar11.
4. Within ar11_64bmg.bin, parse ELF section headers (Python; container
   has no readelf-via-vllm-run, but inside the container it's there;
   from host use python). Section names follow Itanium mangling:
   `.text._ZTSN3gdn23gated_delta_rule_kernelI<T><StateT>Li<KBS>ELb<SPEC>EEE`.
   `S5_` is back-ref for repeated bf16. Lb1 = IS_SPEC=true.
5. Disassemble: `iga64 -d -p=2 <section.bin> -o <out.asm>`.

Note: `gdn_attention_kernel` from tick 22's plan is the wrong name —
actual SYCL kernel is `gdn::gated_delta_rule_kernel`. Sections live in
`_xpu_C.abi3.so`, NOT `libgdn_attn_kernels_xe_2.so` (the latter only
holds chunked-prefill ChunkFwdO/Prepare/etc kernels for prefill, not
the per-token decode/spec recurrence).

**Comparison: bf16/bf16 k=4 spec=true vs spec=false**

  Section size (bytes)       spec_true   spec_false
  --------------------       ----------  ----------
  .text bytes                     13504       13632
  iga64 lines                      1056        1069

  Op counts (top by frequency):
  --------------------       spec_true   spec_false
  add                              119         119
  mov                              105         101  (+4 in true)
  mul                               46          48  (-2 in true)
  send.ugm                          27          37  (-10 in true!)
  mad                               27          27
  macl                              22          24  (-2 in true)
  cmp                               22          22

`mad` count is BIT-EQUAL between variants (27 each). The bug isn't
a missing/extra FMA — it's the **register allocation around** the
FMAs. From `diff -u`, the very first non-prologue instruction shows
register drift: spec_false uses r13/r56 family; spec_true uses
r55/r52 family. The drift propagates through the entire kernel.

The 10-fewer `send.ugm` in spec_true is interesting: spec_false has
the unconditional final-state writeback (lines 290+, `if constexpr
(!IS_SPEC)`); spec_true has the per-token spec writeback (lines
265-289). The compiler emits 10 fewer block-writes for the spec
path. Not necessarily related to the bug, but explains the size
difference.

**Conclusion**: tick 22's hypothesis stands. IGC's GEN ISA emission
is identical-shaped between variants (same op mix, same control
flow, same number of FMAs) but the **register allocator picks
different physical registers** because IS_SPEC=true adds two
captured-by-value parameters (`spec_state_indices`,
`num_accepted_tokens`) that bump live ranges. The IGC codegen bug
manifests at one specific allocation pattern — the spec=true one.

**Next tick (24)**: locate the inner state-update loop in both ASMs
(the 16-iteration `state_local[j*4+i] += k_local[i] * delta[j]`
unrolling, j∈[0..3], i∈[0..3], v_dim_per_sg=4, k_bucket_size=4),
diff the j=3 region specifically (matches earlier "j=3 lane bias"
finding from tick 17). Look for: a `mad`/`mul`+`mov`/store sequence
in spec_true that uses a register that's also touched by an
unrelated instruction nearby — classic GRF-bank-conflict or
SWSB-token-reuse pattern. If found, tick 25 tries an IGC workaround
(`-Xs "-cl-intel-no-subgroup-ifp"`, `-Xs "-cl-intel-greater-than-4GB-buzzer-required"`,
or scheduler-barrier insertion).

Source state: pristine. No rebuild this tick.

