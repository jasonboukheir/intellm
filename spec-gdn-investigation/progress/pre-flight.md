Pre-flight 2026-05-07: dev shell + container OK
(`nix develop ~/Projects/vllm-xpu-kernels`, container `vllm-dev`
auto-creates from `localhost/vllm-xpu-int4-tq:gdn-fix-ccd77bdf4-squashed`).
Replay test collects 2 items
(`tests/kernels/xpu/test_spec_gdn_replay.py::{test_payload_schema,test_sycl_matches_fla}`)
and skips cleanly when `/tmp/spec_gdn_captures` is empty. `flake.nix`
in `vllm-xpu-kernels` is in `.git/info/exclude` — `git add -fN flake.nix`
(intent-to-add only; no content staged) is the standing workaround so
`nix develop` can read it; harmless to leave staged across ticks.

