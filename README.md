# hipbridge

Recognize CUDA kernels, substitute verified AMD implementations, prove it numerically.

**Status: pre-alpha.** Nothing here is verified on AMD hardware yet. Claims in this
README are limited to what the test suite actually exercises.

## Design rules

1. **Never emit a kernel we cannot verify.** An unrecognized kernel returns
   `UNKNOWN` and a report. It does not fall through to a default pattern.
2. **The verifier gates the translator, not the other way round.** `hipbridge.verify`
   is built first and can be used entirely on its own.
3. **Core never imports an extra.** `frontend`, `recognize`, and `analysis` must
   not import `hipbridge.verify` or `hipbridge.kernels`. Enforced by
   `tests/test_layering.py`.

## Install

```bash
pip install hipbridge              # core: frontend + recognize + analysis (libclang only)
pip install hipbridge[kernels]     # + tuned AMD Triton implementations
pip install hipbridge[verify]      # + differential harness (torch, triton)
pip install hipbridge[all]
```

Core installs anywhere. The `kernels` and `verify` extras pull Triton, which
publishes Linux wheels only; the dependency is marked so it degrades instead of
failing the install.

## Layout and the future split

`src/hipbridge/verify/` and `src/hipbridge/kernels/` are kept import-clean so each can
be promoted to its own distribution without a refactor. Promote on evidence
(independent users, independent issues), not on a hunch.
