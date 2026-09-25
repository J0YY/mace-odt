# MACE-ODT research log

## 2026-09-22: lean physical representation hypothesis

### Multi-consumer tangent pilot locked

The shared-interface result showed that the first-branch basis does not preserve
the full model as well as the local radial baseline. The next falsification test
uses both learned energy readouts at the first interaction density. It does not
retrain MACE. It learns only a frozen post-training subspace from 128 label-free
discovery geometries that exclude the existing 64 evaluation geometries.

The hopeful outcome is precise. The nonlinear branch may reveal a lean shared
radial function space that the first linear branch cannot identify alone. If the
rank-64 data-assisted basis beats both existing baselines in held-out force
fidelity without sacrificing energy fidelity, the direction remains worth
pursuing. If it fails at ranks 64 and 80, this tangent construction is rejected.

This pilot is not labeled ODT. An exact ODT-aligned next step would compile the
first readout together with the second-layer immediate consumer bundle, or the
sixteen nonlinear-head preactivations, as a factorized polynomial target.

The first T1 execution stopped at its preregistered fold-coverage gate before
any fidelity evaluation. Both iodine configurations had landed in fold B under
hash parity. The fold assignment was changed using geometry-only species
presence, with rare species assigned first and balanced deterministically. This
change does not use gradients, model errors, or reference labels. One iodine
configuration per fold establishes definedness only. It is not evidence that an
iodine-specific subspace is stable.

### T2 result

The data-assisted two-consumer basis passes the weak feasibility rule on the
frozen 64. At rank 64 it reduces mean force RMSE from `0.00763` to `0.00495
eV/angstrom` against local radial SVD, and reduces mean energy error per atom
from `0.000420` to `0.000127 eV`. It also beats the first-branch global basis.
The corrected paired intervals exclude zero.

The strong rank-64 rule fails. Rank 64 is not yet accurate enough and retains
two thirds of each multiplicity space. Rank 80 reaches `0.000499 eV/angstrom`
and `0.0000171 eV` per atom, but retains five sixths of the space. This is a
positive direction-selection result, not a compact architecture result.

The nonlinear-only basis beats the linear-only basis on force fidelity,
nonlinear-branch fidelity, and final-head preactivations. The balanced basis
beats either alone. This supports the claim that the deeper consumer changes
which radial function directions matter. The summed-energy basis is best for
force fidelity but permits larger cancelling branch errors.

The next method is the exact immediate-consumer coefficient environment. The
data-assisted tangent basis remains a useful baseline. It is not renamed ODT
and receives no coefficient-tail certificate.

### September 24 exact environment update

The exact degree-three immediate-consumer environment now passes its frozen
checkpoint gates. It combines the first scaled readout, the second interaction
message, and the second interaction skip under one tied functional projector.
The largest algebraic or numerical residual is `1.06e-14` with a `5e-9` gate.

The globally trace-balanced spectrum retains `99.7706` percent of its trace at
rank 16, `99.9992` percent at rank 48, and more than `99.9999998` percent at
rank 80. This makes the hypothesis worth testing but does not establish model
fidelity. The frozen T4 evaluation is now running on 64 held-out structures.
Its primary method is the trace-balanced exact basis. Raw equal weighting,
single-consumer bases, local radial SVD, the earlier exact first-branch basis,
the total-energy-gradient basis, reverse order, and five random bases are fixed
controls. No rank or basis is selected from evaluation labels.

### Origin

David Olloqui relayed a research hope discussed with Ward. It would be valuable
if the large internal MACE representation could be replaced with a much leaner
weight, function, or parameter space whose coordinates have a direct physical
organization.

### Logged hypothesis

A pretrained MACE potential may contain a compact, gauge-stable basis of
element-pair radial and angular functions that can replace part of its hidden
representation without retraining while preserving relative energies and
forces.

The first target is the bounded-order polynomial branch feeding the first
linear energy readout. For central species `z`, neighbor species `z'`, angular
degree `l`, and retained mode `k`, the desired coordinates have the form

```text
phi[k, z, z', l](r, Omega) = f[k, z, z', l](r) Y[l](Omega)
```

The retained functions are paired with compact interaction cores for the
linear, quadratic, and cubic density terms.

### Why this is more than ordinary compression

The retained coordinates must satisfy four requirements.

1. They are selected using the composed energy computation rather than one
   local matrix.
2. They are represented as functions of species and geometry rather than raw
   channel vectors.
3. Their projectors and interventions transport correctly under allowed hidden
   coordinate changes.
4. Their value is tested with energy differences and forces rather than only a
   coefficient spectrum.

### Two-stage research program

**Stage 1, discovery.** Extract a lean physical function space from frozen
checkpoints. No model training is required. This stage tests whether global ODT
selection improves the rank needed for physical fidelity.

**Stage 2, architecture.** Determine whether modes recur across seeds, model
sizes, and checkpoints. If they do, approximate them with a compact spline or
analytic radial family and test them as a trainable MACE parameterization.

Stage 2 is not assumed to work. A checkpoint-specific compact basis can still
be useful for model analysis and compiled inference.

### Scope and claim boundary

An exact or compressed first-readout branch is not a compressed full MACE
model. A branch-only evaluator may leave the original nonlinear branch in
place, so it may not improve end-to-end runtime. Full representation
replacement requires adapting every consumer of the selected interface.

A visually simple radial mode is not automatically a chemical mechanism. The
reduced interaction core and controlled interventions must be reported with
the function image.

### First decision criterion

Continue to the full rank study only if the untruncated compiler reproduces the
checkpoint's first branch and coordinate derivatives at the calibrated
floating-point floor.

The first positive scientific result would be a lower retained rank than local
weight or radial-function baselines at the same preregistered relative-energy
and force tolerance.

### Implementation status

- Synthetic algebra checks are complete.
- Independent finite-dimensional audit checks are complete.
- Athena access and Slurm partitions were verified.
- No MACE environment or MACE checkpoint was present before this implementation.
- A work-mounted Athena environment was created with MACE 0.3.16 and PyTorch
  2.6.0 CPU.
- E1 completed on Slurm job 399950 using the official MACE-OFF23 small
  checkpoint. All checkpoint smoke tests passed.
- E2 completed on Slurm job 399952. The serialized cubic scalar coupling map
  has 23 stored paths and an eight-dimensional supported repeated-slot image in
  both layers.
- The live exact path quotient passed on Slurm job 399969. Replacing both
  product contractions changed tested forces by at most `8.88e-16
  eV/angstrom`.
- The registered radial interface passed on Slurm job 399973. Its multineighbor
  density, first-branch energy, and force reconstruction errors were at the
  floating-point floor.
- The exact first-branch compiler passed on Slurm job 399988. It includes the
  post-product map, linear readout, and checkpoint scale.
- Equal-weight and coefficient-balanced global environments passed on Slurm
  jobs 399993 and 399994. Every slot is contracted explicitly and all
  cross-path terms are retained.

### First real-checkpoint evidence

The checkpoint audit found 694,320 parameters, two interaction blocks, 96
channels in each scalar product output, ten supported elements, and a 4.5
angstrom cutoff. The first readout is linear with 96 weights. The second
readout is a nonlinear 96 to 16 to 1 head.

The energy and force evaluator passed translation, rotation, and atom-order
checks on methane and water. The methane force finite-difference discrepancy
was approximately `2.05e-8 eV/angstrom`.

For both product layers, the order-one, order-two, and order-three path maps
have stored-to-supported dimensions `1 to 1`, `4 to 4`, and `23 to 8`. A
random cubic null perturbation produced a response norm of approximately
`2.50e-16`, while a retained perturbation produced a response norm of
approximately `4.73` in the same synthetic check.

This is a successful reproduction of exact architectural redundancy. It is not
by itself evidence of a smaller end-to-end potential.

### 2026-09-22: first global coefficient spectra

The exact branch compiler matches random aggregate features within
`6.54e-13 eV`. On methane and water, branch energy errors are at most
`8.88e-16 eV` and force errors are at most `9.99e-16 eV/angstrom`.

The global environment obeys the required rotational block form. Across all
central species, the relative off-block norm is below `8.2e-19`, magnetic-copy
deviation is below `4.5e-17`, and the occurrence-weighted trace identity holds
within `1.4e-15` relative error.

Equal order weights are dominated by the linear coefficient norm. A second,
predeclared formula sets each order weight to the inverse full coefficient norm
for that species and order. Under this coefficient-balanced convention, a
uniform multiplicity rank of 16 per irrep leaves about `0.0033` to `0.020`
relative squared coefficient error across central species. Rank 32 leaves about
`0.00024` to `0.00125`.

This is preliminary evidence of composed coefficient low rank. It is not yet a
held-out force-fidelity result. The next decision requires a frozen molecular
evaluation set and matched global, local radial, local weight, and random
baselines.

### 2026-09-22: frozen official test subset

The official MACE-OFF23 test archive was downloaded from the Cambridge
repository record `10.17863/CAM.107498`. The archive bitstream identifier is
`cb8351dd-f09c-413f-921c-67a702a7f0c5`, and its verified MD5 checksum is
`bf1f7c7ff1714e6ac6af2cdc642a130a`.

The extracted test file contains 50,195 configurations across seven declared
configuration types. Before evaluating any projected errors, a 64-configuration
subset was frozen by metadata strata and SHA-256 scores of the seed and original
frame index. The extracted XYZ SHA-256 is
`c65a7fcf9140fb127ccc6cc7fce243ab5052665a8741516cde143592a319745e`.

The held-out branch-fidelity run compares the global environment, local radial
SVD, and three canonical random seeds at ranks 16, 32, 48, 64, and 80. It
reports paired bootstrap intervals for local minus global error. The subspaces
remain weight-derived and do not use these configurations for fitting.

### 2026-09-22: first held-out rank-fidelity result

Athena job 400007 evaluated the frozen 64-configuration subset. The global
environment has lower mean per-atom branch energy error and lower mean branch
force RMSE than local radial SVD at every tested rank.

At multiplicity rank 64 per irrep, the global mean force RMSE is
`0.00319 eV/angstrom`, compared with `0.00921 eV/angstrom` for local radial
SVD. At rank 80, the values are `0.000306` and `0.00120 eV/angstrom`.

Paired configuration-level bootstrap intervals for local minus global mean
force error exclude zero at every tested rank. At rank 64 the mean difference
is `0.00602 eV/angstrom`, with a 95 percent interval from `0.00533` to
`0.00660`. At rank 80 the mean difference is `0.000895 eV/angstrom`, with an
interval from `0.000832` to `0.000961`.

This clears the first branch-level feasibility question. It does not yet show
that the full model can use the same reduced representation, that runtime is
lower, or that the retained functions have a stable chemical interpretation.

### 2026-09-22: shared-interface boundary result

The same encoder and decoder were inserted before product block zero, so the
first readout and every later consumer received projected features. This tests
the full frozen model without retraining.

The first-branch global advantage does not transfer automatically. At rank 64,
mean shared-model force RMSE is `0.00925 eV/angstrom` for the first-branch
global basis and `0.00763 eV/angstrom` for local radial SVD. At rank 80, the
values are `0.00174` and `0.000913 eV/angstrom`.

This is a useful negative boundary rather than a failure of the branch result.
It shows that the nonlinear consumer uses directions that the first-readout
environment discounts. A full-model method must add the later consumer to the
environment objective or solve an explicitly multi-consumer projection
problem. Reusing the first-branch score is not sufficient evidence for general
MACE compression.

### 2026-09-24: exact immediate-consumer held-out verdict

The frozen 64-structure evaluation completed and its rank-96 replay gate
passed. The trace-balanced exact method reached force RMSE `0.008060 eV/A` and
energy error `0.000568 eV/atom` at rank 64. It was better than the prior exact
first-branch basis, but worse than local radial SVD and the
total-energy-gradient basis. At rank 80 it met the absolute error targets, with
force RMSE `0.000986 eV/A` and energy error `0.0000429 eV/atom`, but it still
lost both relative comparisons.

The preregistered outcome is no-go for this immediate-consumer objective and
frozen measure. The broader lean-space hypothesis remains open. The experiment
shows that the exact local coefficient tail is not by itself a reliable proxy
for whole-model force error. The next credible method should combine exact
equivariant consumer structure with global energy or force sensitivity.

### 2026-09-24: objective pivot after T4

The failure is now localized more precisely. T4 does not reject generalized
ODT or a compact functional representation. It rejects a geometry-free
immediate-consumer coefficient norm as the final ranking rule for the nonlinear
full-model projector.

The exact structural environment remains useful as a gauge-stable physical
coordinate system and as a regularizer. It has no established bound on the
finite full-model energy and force error. The next primary objective is the
full-graph tangent energy and force loss under the actual tied projector. The
definitive empirical target is the finite frozen-model energy and force loss.

This pivot is farther from the literal Dooms Algorithms 1 through 3 because it
uses molecular geometries, full-model derivatives, and a joint Grassmann
optimization rather than one weight-only bond eigendecomposition. It remains
close to the central Dooms principle because the subspace is selected by
pulling the rest of the computation back to a registered internal cut.

The next study will include a Dooms-proximity ablation. A global-adjoint block
environment with spectral truncation will be compared directly with the full
tied tangent objective and finite projector optimization. This will determine
whether the departure from the original algorithm is necessary. The frozen
mathematical objective and experiment order are recorded in
`GRAPH_PULLBACK_OBJECTIVE.md`.

The frozen ladder now separates six objects. D0 is the structural coefficient
environment. G0 is the existing total-energy-gradient baseline. D1-E is a
spectral activation-aware energy pullback. D1-EF adds force sensitivity while
retaining spectral truncation. D2 optimizes the coherent tangent objective
jointly. D3 optimizes the finite frozen-model loss.

A full claim about the minimum departure from Dooms requires D1-E, D1-EF, D2,
and D3 all to complete under matched discovery folds. If one is infeasible, the
result will be described only as the earliest supported method among completed
rungs. D1-EF succeeding without a D2 or D3 gain would support a close spectral
extension. A D2 gain would show that the separable spectral upper bound is
insufficient. A result requiring D3 would support structured frozen compression
rather than a classical ODT generalization.
