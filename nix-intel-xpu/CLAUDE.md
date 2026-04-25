# nix-intel-xpu

`level-zero` and `intel-compute-runtime` are in nixpkgs — use those directly.

Intel DPC++ compiler (icpx) cannot be cleanly packaged in nix yet — use the `oneapi-env` FHS wrapper.
