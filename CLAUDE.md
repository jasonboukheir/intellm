# intellm

Monorepo. Each sub-project has its own `flake.nix` + `.envrc` — `cd` in and direnv activates the shell.

SYCL kernel compilation requires Intel DPC++, which isn't nix-packaged. Enter the FHS env first (`nix run nix-intel-xpu#oneapi-env`), then `icpx -fsycl` works inside that shell.
