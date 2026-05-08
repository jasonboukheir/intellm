#!/usr/bin/env bash
# Recipe presets for W4A16 quantization of Qwen3.6-35B-A3B on Battlemage B70 30 GB.
# Tuned from per-knob smokes on Qwen3-0.6B 2026-05-02 (see `help`).
set -euo pipefail

MODEL="${AUTOROUND_QWEN_MODEL:-Qwen/Qwen3.6-35B-A3B}"
RECIPE_BIN="${AUTOROUND_QWEN_RECIPE_BIN:-auto-round}"
SEQLEN="${AUTOROUND_QWEN_SEQLEN:-2048}"

usage() {
    cat <<'EOF'
auto-round-qwen-3-6-35b-a3b — W4A16 quantization presets for Qwen3.6-35B-A3B on B70

Usage:
  auto-round-qwen-3-6-35b-a3b safe        [extra auto-round flags...]
  auto-round-qwen-3-6-35b-a3b aggressive  [extra auto-round flags...]
  auto-round-qwen-3-6-35b-a3b help

Presets (both: --scheme W4A16 --format auto_round, recipe=auto-round 200 iters, seqlen 2048):

  safe          bs=4 ga=2  --low_gpu_mem_usage OFF
                est ~5h. Peak VRAM not directly measured on 35B — extrapolating
                from light baseline (19.0 GB, low_gpu_mem ON) × 1.53 (the +53% from
                dropping low_gpu_mem) gives ~29 GB. Likely tight. Smoke first if
                running for the first time on a new model size.

  aggressive    bs=8 ga=1  --low_gpu_mem_usage OFF
                est ~4h. **MEASURED on 35B 2026-05-02: peak VRAM 29.36 GB** (single
                block, then stable at block 1). Per-block time 104s vs 155s light
                baseline = 1.49× speedup, matching the 1.43× combined prediction.
                FITS but with only 0.64 GB headroom under the 30 GB cap — too tight
                for unattended overnight runs. Use safe instead unless you can babysit.

Environment overrides:
  AUTOROUND_QWEN_MODEL         model id (default: Qwen/Qwen3.6-35B-A3B)
  AUTOROUND_QWEN_RECIPE_BIN    auto-round | auto-round-light | auto-round-best
                               (default: auto-round; switch to -light for ~1.5h smoke)
  AUTOROUND_QWEN_SEQLEN        calibration seqlen (default: 2048)
  AUTOROUND_OUTPUT_DIR         output dir base (default: $PWD/output)

Per-knob measurements (Qwen3-0.6B, B70 idle, auto-round-light 50 iters):

  | run | bs × ga | seqlen | low_gpu_mem | compile | tuning  | s/it | VRAM    | speedup |
  |-----|---------|--------|-------------|---------|---------|------|---------|---------|
  | A   | 4 × 2   | 1024   | on          | -       | 126.72s | 4.52 | 1.47 GB | 1.000×  |
  | B   | 1 × 8   | 1024   | on          | -       | 191.93s | 6.84 | 0.63 GB | 0.660×  |
  | C   | 4 × 2   | 2048   | on          | -       | 252.72s | 9.02 | 4.82 GB | 0.502×  |
  | D   | 4 × 2   | 1024   | on          | on      | 198.97s | 7.09 | 1.28 GB | 0.637×  |
  | E   | 4 × 2   | 1024   | OFF         | -       |  97.89s | 3.48 | 2.25 GB | 1.295×  |
  | F   | 8 × 1   | 1024   | on          | -       | 115.73s | 4.13 | 2.82 GB | 1.095×  |
  | G   | 8 × 1   | 1024   | OFF         | -       |  88.49s | 3.15 | 3.38 GB | 1.432×  |

  (safe = E-style on 35B; aggressive = G-style on 35B)

Per-knob multipliers:
  bs=4 ga=2 vs bs=1 ga=8 (matched tokens):  1.51× win, +30% VRAM     ← biggest lever
  drop --low_gpu_mem_usage:                 1.30× win, +53% VRAM     ← safe preset uses
  bs=8 ga=1 vs bs=4 ga=2 (matched tokens):  1.10× win, +92% VRAM     ← diminishing
  drop low_gpu_mem + bs=8 (compose):        1.43× win (1.01× compose efficiency)
  --enable_torch_compile (dense):           1.57× LOSS               ← do NOT use on dense
                                            (sign open on MoE; would need single-block 35B
                                             smoke to confirm; until then, leave off)

Default-recipe extrapolations from 18h baseline (auto-round 200 iters seqlen 2048 bs=1 ga=8):
  + bs=4 ga=2                               12h
  + drop low_gpu_mem  (= safe)               5h   ← 3.6× win, ~29 GB est (tight)
  + bs=8 ga=1 also   (= aggressive)          4h   ← 4.5× win, 29.36 GB MEASURED
                                                   (only 0.64 GB headroom — tight)

Always KL-test the result against BF16 before committing to a vLLM deployment.
EOF
}

run_preset() {
    local preset="$1"; shift
    local bs ga
    case "$preset" in
        safe)       bs=4; ga=2 ;;
        aggressive) bs=8; ga=1 ;;
        *) echo "error: unknown preset '$preset' (expected: safe|aggressive)" >&2; exit 1 ;;
    esac

    echo ">>> auto-round-qwen-3-6-35b-a3b $preset"
    echo ">>> model=$MODEL recipe=$RECIPE_BIN bs=$bs ga=$ga seqlen=$SEQLEN low_gpu_mem=OFF"
    echo ""

    # Use `autoround run --` to bypass the wrapper's hard-coded --low_gpu_mem_usage,
    # --batch_size, --gradient_accumulate_steps, and --seqlen defaults. We pass the
    # full auto-round CLI ourselves.
    exec autoround run -- "$RECIPE_BIN" \
        --model "$MODEL" \
        --scheme W4A16 \
        --format auto_round \
        --device 0 \
        --batch_size "$bs" \
        --gradient_accumulate_steps "$ga" \
        --seqlen "$SEQLEN" \
        --output_dir /output \
        "$@"
}

main() {
    local sub="${1:-help}"
    case "$sub" in
        help|-h|--help) usage ;;
        safe|aggressive) shift; run_preset "$sub" "$@" ;;
        *) echo "error: unknown subcommand '$sub'" >&2; usage >&2; exit 1 ;;
    esac
}

main "$@"
