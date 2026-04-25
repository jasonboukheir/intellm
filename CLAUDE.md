# intellm

Monorepo. Each sub-project has its own `flake.nix` + `.envrc` — `cd` in and direnv activates the shell.

SYCL kernels need Intel DPC++ (`icpx -fsycl`), which lives in the FHS env: `nix run nix-intel-xpu#oneapi-env`.
