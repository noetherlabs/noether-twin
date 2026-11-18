# Registration or resolution?

The completed [development report](../results/registration_development_v1/README.md) retains all 51 units, unresolved fits and computation costs. This page specifies the frozen acquisition and interpretation rules.

This development diagnostic asks whether a fixed calibration exposure budget
improves predictions more than increasing pupil-grid resolution. It fits one
mirror–pupil translation from focal-plane images with a known flat incoming
wavefront. It does not select a controller, establish a global fit, or measure
closed-loop or hardware performance. The earlier static action ties and failed
development discrimination remain unchanged; this protocol does not promote
those cohorts into held-out evaluation.

## Physical experiment

The existing 8 m obscured pupil, 551 nm wavelength, 64 m focal length, 65×65 focal
pixels and 5×5 Gaussian mirror remain fixed. Mirror coupling is 0.45, cutoff is
three projected pitches, and projected pitch is 1.6 m. Only actuator centres move
along x: the three fixed offsets are −0.10, 0 and +0.10 pitch. These are simulation
sensitivity challenges, not measured instrument uncertainties. The pupil,
influence width, y alignment, rotation, flux, diversity and command gain remain
known. All candidate grids sample the same continuous physical mirror model.

HCIPy independently constructs the plant at 512 samples and the refinement
diagnostic at 1024. dLux uses 64, 128 and 256 samples. Physical mirror surface
commands receive the same double-pass OPD convention. Pixel power is the existing
fixed-grid quadrature; finite-image flux is never renormalized.

| Role | Row-major actuator | Surface command | Labelled diversity OPD | Images |
| --- | --- | --- | --- | --- |
| Calibration | 12 and 7 | ±10 nm | ±30 nm | 8 |
| Unseen-pattern scoring | 18 | ±10 nm | ±30 nm | 4 |

Each diversity pair receives one million incident photons, divided equally
between exposures, plus 0.1 background count per pixel per exposure. Every noisy
unit therefore uses four million calibration photons and two million scoring
photons. The same calibration observations are shared across every candidate.
The fully labelled spatial images preserve sign information; no radial averaging
or fitted photometric scale enters the estimator.

Known-flat illumination is an explicit restriction. An instrument study needs
independent support for that state or a matched incoming-aberration nuisance
model. A measured optical response alone does not identify physical DM gain.

## Estimator and baseline

Every grid minimizes the same joint Poisson negative log-likelihood, excluding
the parameter-independent factorial and dividing by four million calibration
photons. The sole parameter is x registration in the box [−0.20,+0.20] pitch.
Projected scalar secant steps use Armijo backtracking with coefficient 10⁻⁴,
halving step sizes and initial inverse curvature one. Nonpositive secant curvature
resets that inverse to one. Starts are −0.20, −0.10, 0, +0.10 and +0.20 pitch;
clip and deduplicate before counting distinct starts.

Each start has at most 64 value-and-gradient calls, including rejected steps and
confirmations. Local stationarity requires projected gradient at most 10⁻⁶,
then two unchanged-point confirmations with objective change at most 10⁻¹⁰ and
the same gradient condition. An initially stationary point need not move.
All starts must finish stationary. At least two distinct starts must reach the
best objective within 10⁻⁸, and near-best endpoints must span at most 10⁻⁴ pitch.
This is a declared local-resolution convention, not global uniqueness or a
metrological accuracy guarantee. Constrained boundary solutions are labelled.

Flat, separated near-best, exhausted or underflowed fits remain unresolved and
predict with nominal zero registration. Their losses and fit costs stay in every
comparison. Nonfinite outputs and backend exceptions are retained and stop the
attempt; they cannot become successful nominal fallbacks.

Each grid supplies nominal and estimated-registration predictions. Fixed 256
with the identical estimator is the matched baseline for a lower-grid cost claim.
An evaluator-only model uses the true registration at each grid to diagnose
discretization. It is not deployable and need not minimize predictive loss on a
biased finite grid: a fitted shift can compensate another modelling error.

## Separation and outcomes

There are three deterministic expected-mean diagnostics and 16 independent
Poisson realizations per offset: 48 noisy units at only three physical states.
Calibration and scoring use distinct frozen seed namespaces. Expected means
remain real-valued and use expected Poisson KL; they are never rounded into
synthetic integer observations. This count is a bounded development budget,
not a power calculation or a population of independent instruments.

```text
shared calibration images -> grid-specific fit or explicit fallback
  -> every nominal/fitted prediction and fit receipt frozen
  -> scoring images, reference refinement and evaluator-only models
  -> all paired losses retained, stratified by physical offset
```

All 153 fits and 306 nominal/estimated prediction blocks freeze before any
scoring-reference, scoring-noise or true-registration prediction is acquired.
The estimator receives calibration data and its candidate parameter only. The
schedule and fixed offsets are declared; this separation is a trusted-code
contract, not a sandbox against an adversarial estimator.

The primary descriptive contrast is estimated128 minus nominal256 in unseen-
pattern Poisson loss per incident scoring photon. It compares two operational
choices, not an isolated calibration or resolution effect. The report also keeps
estimated128 minus estimated256, estimated-minus-nominal at each grid,
estimated-minus-true-registration diagnostics, zero-offset harm, registration
error, abstention and full fit costs. Negative loss differences favour the first
arm. No success-only row filtering or pooled winner is reported.

The identical frozen predictions are also scored on 512- and 1024-grid expected
means. Both expected contrasts and their absolute difference are retained. A
reversed, zero or smaller-than-refinement contrast is labelled unresolved; larger
contrasts remain descriptive only. This sensitivity diagnostic is neither a
continuum error bound nor a statistical confidence interval. No method-superiority
gate or decision margin is established here. Agreement between grids supports a
calibration-limited interpretation only if calibration improves prediction beyond
the observed sensitivity; agreement alone establishes no limiting mechanism.

## Reproducibility and cost

`noether_optics.bench.twin_registration.protocol()` declares the complete
schedule, seeds, solver settings and limits. A separately admitted environment
can run `python -m noether_optics.bench.twin_registration NEW_OUTPUT_DIRECTORY`
with JAX float64 on one CPU worker. No new dependencies are required.

The maximum ledger is 48,960 joint objective/gradient calls, 72 independent
reference images and 459 candidate four-image prediction blocks. Six candidate
compilations are timed separately; all rejected proposals and confirmations
contribute to complete fit costs. Timing includes synchronized host outputs,
and fit wall time includes evidence I/O. These are not real-time guarantees or an
equal-resource controller comparison.

Use fresh source/runtime/storage admission and an external single-worker
supervisor: prospective 30 minute wall time, 4 GiB process-group RSS, 512 MiB
logical packet, 1 GiB allocated packet and 256 MiB temporary allowance, preserving
the programme's admitted disk reserve. These are finite operational limits, not
measured cost or proof that the experiment will fit. No automatic limit increase,
retry or silent cohort reduction is permitted.

Every attempt uses a fresh directory. Dispatches precede numerical callbacks;
raw arrays, failures, fit traces, prediction receipts and timing remain immutable.
A complete manifest requires full declared coverage and unchanged source hashes.
Runtime/owned-process finalization also requires the external supervisor. A
complete execution establishes neither a scientific win nor promotion to a new
selection/evaluation cohort. Delayed closed-loop consequence, equal measured
decision availability and independent instrument data remain separate studies.
