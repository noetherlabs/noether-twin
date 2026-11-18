# Noether Twin

Fixed-budget optical calibration with explicit fallback and retained evidence.

The current study asks whether eight focal-plane calibration images can improve a model more than increasing its pupil-grid resolution. It compares an independently implemented HCIPy optical plant with differentiable dLux models, and preserves every fit, prediction and abstention.

**Result:** calibration reduced same-grid mean predictive loss in two shifted states, but calibrated 128-sample predictions had higher mean loss than nominal 256-sample predictions in every state. At 128 samples, 18 of 32 shifted noisy fits used nominal fallback. This known-flat simulation establishes neither hardware performance nor a closed-loop control benefit.

[Read the study](results/registration_development_v1/README.md) · [Protocol](docs/twin-registration.md) · [Complete derived evidence](results/registration_development_v1/summary.json)

## Included

- Scalar mirror–pupil registration from labelled phase-diversity images.
- A fixed five-start estimator with charged proposals, confirmations and nominal fallback.
- Independent optical references, finite-difference checks and complete injected packet tests.
- A 51-unit development report with all 153 fits, three figures and reproducible plotting code.

The original estimator and reported outcomes remain unchanged. All 63 unresolved fits satisfied stationarity but exceeded the declared endpoint-agreement tolerance; this numerical disagreement is not evidence of physical nonidentifiability.

## Install and check

Use Python 3.11 and a fresh repository-local environment. The distribution and import names remain `noether-optics` and `noether_optics` to preserve the acquisition source; keep this environment separate from other Noether repositories using those names.

```bash
uv sync --locked --python 3.11.15 --extra jax --extra dev
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 \
  uv run --no-sync pytest tests/bench/test_twin_registration.py tests/twin/test_static_mirror.py -q
uv run --no-sync python scripts/plot_registration_development.py --output figures
```

The [correctness workflow](.github/workflows/registration.yml) uses a pinned Python 3.11.15 environment and the committed dependency lock on relevant pushes and pull requests. Scientific tests and launcher tests run in separate interpreters: Linux counts memory inherited before `exec` in a child's lifetime peak, so resource-limit fixtures must not inherit a populated optical-model process. Every test remains included, with separate JUnit reports. The original implementation passed 128 local checks; the retained-data audit and figure review are separate evidence.

## Reproduce the study

The protocol specifies the optical configuration, observation seeds, call budgets and resource limits. The included launcher checks the source and pinned runtime, then runs the frozen study with one CPU worker and JAX float64:

```bash
uv run --no-sync python scripts/run_registration.py NEW_OUTPUT_ROOT --min-free-gib 4
```

The output root must be fresh. Its `packet/` contains the study, `runtime/` retains logs and caches, and `receipt.json` records completion or the reason execution stopped. The launcher samples wall time (30 minutes), process-group memory (4 GiB), packet size (512 MiB logical / 1 GiB allocated), runtime output (256 MiB), and the requested free-space reserve. At exit it also checks the owned child's kernel-reported peak memory, including pre-`exec` memory, against the same 4 GiB limit; earlier unrelated children do not contribute. Add `--reserve-path PATH` for each additional backing filesystem, including a disk-image backing directory. Free space on a virtual volume alone need not measure its backing store.

Failed runs retain their partial output and require a new root for retry. The checkout lock coordinates this launcher only. These sampled controls support reproduction of reviewed code; they are not a security sandbox, a real-time guarantee or scientific qualification. Source archives run without discovering a parent Git repository. A new execution receives its own timings and artifact identities.

The original 131 MB raw acquisition is retained separately; this repository includes complete per-unit derived evidence and its source/audit hashes, with no raw download.

## Scope

The experiment assumes a known-flat incoming wavefront, fixed flux/gain/diversity and three mirror–pupil offsets. It contains 16 noisy realizations per state plus separate mean-image diagnostics. Component timings include evidence I/O and are not control-loop deadlines. Instrument calibration drift, uncertainty in recovered physical parameters and controller consequences need separate evidence.

See [source provenance](docs/source-provenance.md) for the exact mapping to the acquisition implementation. Code, documentation and included derived artifacts use [Apache-2.0](LICENSE).
