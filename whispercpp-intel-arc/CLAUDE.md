# whispercpp-intel-arc

whisper.cpp + SYCL backend on Intel Arc Pro B70 (Battlemage / Xe2).
Sibling of `../llamacpp-intel-arc/`; same intel/vllm:0.17.0-xpu runtime
image, same vendoring pattern into `~/.config/nix/pkgs/`.

Goal: serve `whisper-large-v3-turbo` for the brutus STT path. CPU
faster-whisper hits ~1-2x realtime on this box; SYCL on B70 should
land at ~5-10x.

## Build

```
scripts/build-aicss.sh
~/.config/nix/scripts/refresh-whispercpp-binary.sh
sudo nixos-rebuild switch --flake ~/.config/nix#brutus
```

## Known gotchas (inherited from sibling)

- Don't pass `GGML_SYCL_F16=ON` — corrupts weights on bmg_g21 unless
  `GGML_SYCL_DISABLE_OPT=1` at runtime. Build script intentionally
  leaves F16 off. Tracking:
  https://github.com/ggml-org/llama.cpp/issues/21893
- F32 SYCL on whisper is fine — encoder/decoder are small enough that
  the F16 perf delta doesn't matter.

## References

- whisper.cpp SYCL doc: https://github.com/ggml-org/whisper.cpp/blob/master/docs/sycl.md
- whisper-server README: https://github.com/ggml-org/whisper.cpp/tree/master/examples/server
- GGML model files: https://huggingface.co/ggerganov/whisper.cpp
