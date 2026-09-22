# MACE-ODT experiment map

## Decision structure

```text
environment and checkpoint provenance
              |
              v
exact checkpoint forward and symmetry tests
              |
              v
exact path quotient and radial reconstruction
              |
              v
untruncated first-branch compiler
              |
              v
global operators and tiny dense oracles
              |
              v
frozen rank ladder against matched baselines
              |
              +--------> physical mode cards and interventions
              |
              +--------> compiled branch cost and full-interface feasibility
```

No spectrum is interpreted before the exactness gates above it pass.

## Current status

| Experiment | Status | Evidence |
|---|---|---|
| E0 environment and provenance | Passed | Athena manifest under `results` |
| E1 checkpoint architecture audit | Passed | Slurm job 399950 |
| E2 repeated-slot path map audit | Passed | Slurm job 399952 |
| E2 full converted-checkpoint integration | Passed | Slurm job 399969 |
| E3 radial reconstruction | Passed | Slurm job 399973 |
| E4 first-readout compiler | Passed | Slurm job 399988 |
| E5 global mixed-order environment | Passed | Slurm jobs 399993 and 399994 |
| E6 diagnostic rank ladder | Passed | Slurm job 399999, smoke geometries only |
| E7 frozen held-out rank ladder | Passed | Athena job 400007 on 64 frozen official test configurations |
| E8 shared-interface fidelity | Passed | Athena jobs 400013 and 400023 on the frozen subset |
| E9 and E10 | Not started | E8 requires a multi-consumer method before a full-model claim |

## Common experimental rules

- Use frozen released checkpoints in the primary study.
- Keep discovery configurations separate from the final evaluation set.
- Preserve complete O(3) multiplets.
- State every radial, angular, species, order, and output metric.
- Report exact quotient dimension separately from learned low rank.
- Compare methods through the same evaluator at the same retained rank.
- Report original-model fidelity separately from reference-label accuracy.
- Treat degenerate eigenspaces as subspaces rather than named individual modes.
- Record checkpoint, source, dependency, configuration, and output hashes.

## E0: environment and provenance

**Question:** Can every result be tied to one executable environment and one
checkpoint?

**Implementation:** `python -m mace_odt.cli.environment_audit`

**Required output:** Python, PyTorch, e3nn, MACE, ASE, NumPy, platform, Slurm,
CUDA, Git state, and configuration hashes.

**Gate:** The manifest is complete and the selected checkpoint file has a
SHA-256 digest.

## E1: checkpoint architecture audit

**Question:** What computation is actually serialized in the released model?

**Implementation:** `python -m mace_odt.cli.checkpoint_audit`

**Required output:** Module tree, tensor shapes, parameter counts, interaction
blocks, product blocks, readouts, radial modules, atomic numbers, cutoff,
correlation metadata, and checkpoint digest.

**Controls:** Run deterministic energy and force smoke tests. Check translation,
rotation, atom-order covariance, and a finite-difference force component.

**Gate:** All smoke tests pass at a tolerance calibrated to float64 CPU
inference. Any failure is investigated before internal extraction.

## E2: exact coupling-path quotient

**Question:** Which stored correlation paths are exactly unable to reach the
selected output representation?

**Implementation:** Reconstruct each symmetrized Clebsch-Gordan path map from
the pinned module graph. Compute its supported row space and null space by
irrep, species, layer, and correlation order.

**Required output:** A table containing stored dimension, supported dimension,
null dimension, singular values, numerical rank threshold, and reconstruction
residual.

**Controls:** Perturb a null vector and a retained vector. The null perturbation
must not change the composed output. The retained perturbation normally should.

**Gate:** The converted and original path maps agree at the numerical floor.

## E3: radial function reconstruction

**Question:** Can the selected native interface be represented exactly as
element-pair radial functions with correct derivatives?

**Implementation:** Evaluate every learned radial channel over a declared
interval. Construct the overlap metric by converged quadrature. Restrict to its
supported space and whiten it with a supported factorization.

**Required output:** Radial curves, derivative curves, metric spectrum,
supported dimension, quadrature convergence, and native-to-functional maps.

**Controls:** Reconstruct the native interface on multineighbor environments.
Test changes of species, radius, orientation, neighbor order, and neighbor
count.

**Gate:** Values and coordinate derivatives match the native module within the
calibrated numerical floor.

## E4: exact first-readout compiler

**Question:** Can the complete first linear-energy branch be written as
symmetric coefficient tensors `K1`, `K2`, and `K3` in the whitened function
space?

**Implementation:** Compose the radial interface, angular basis, symmetrized
products, exact path quotient, species maps, residual terms, scaling, and first
linear readout.

**Required output:** Factorized and tiny dense coefficient forms with metadata
for every species and O(3) block.

**Controls:** Compare the native branch and compiled branch on synthetic local
environments and evaluation-only molecular geometries. Compare energies,
coordinate derivatives, and body-order inclusion-exclusion identities.

**Gate:** The untruncated compiler matches before any decomposition is run.

## E5: global operator construction

**Question:** Does downstream composition alter the important subspace relative
to local decompositions?

**Implementation:** Build mixed-order marginal operators from `K1`, `K2`, and
`K3`. Keep order weights explicit. Use matrix-free products for production and
dense contractions on tiny supported spaces as an oracle.

**Required output:** Eigenvalues, stable eigenspaces, residual bounds, and
agreement between factorized and dense implementations.

**Controls:** Test compensated gauges, exact null additions, random orthogonal
changes in whitened space, and near-degenerate synthetic cases.

**Gate:** Operator actions, spectra, projectors, and exported functions obey
their derived transport laws.

## E6: descriptive spectrum comparison

**Question:** Is any apparent compactness global rather than a trivial endpoint
or architecture ceiling?

**Methods:** Global MACE-ODT, local weight SVD, local radial-function SVD, exact
quotient only, canonical random subspaces, activation-aware PCA, and an
architecture-matched random or untrained control when available.

**Required output:** Spectra, cumulative mass, effective dimensions, subspace
overlaps, exact support ceilings, and confidence intervals across species and
irreps.

**Gate:** This experiment is descriptive. It cannot establish compression by
itself.

## E7: frozen rank-fidelity ladder

**Question:** Which method needs the fewest retained functions to preserve the
first branch?

**Metrics:** Coefficient error, branch energy error, relative-energy error,
force error, torsional profile error, difficult-geometry error, and the proved
structural upper bound.

**Required output:** Fidelity versus rank curves using one frozen set of ranks,
metrics, and configurations for all methods.

**Primary decision:** Compare the retained rank required to reach a declared
physical fidelity tolerance. Do not choose a rank from the final test curves.

**Negative result:** If global selection does not beat local methods, report
the architectural quotient and the absence of additional useful composed low
rank.

## E8: shared-interface intervention

**Question:** Does a subspace sufficient for the first branch preserve other
consumers of the same hidden representation?

**Evaluators:** One branch-only evaluator replaces only the linear branch. One
shared-interface evaluator inserts the projector before every native consumer.

**Required output:** Both fidelity curves and their difference.

**Interpretation:** A large gap shows that the nonlinear branch uses discarded
information. It does not invalidate the branch result.

## E9: physical interpretation

**Question:** Do retained subspaces support controlled and predictive accounts
of model behavior?

**Required mode card:** Element-pair radial images, angular multiplet, parity,
body-order content, reduced interaction core, gauge tests, metric sensitivity,
spectral gap, and intervention definition.

**Controls:** Match edit support and amplitude. Compare declared encoder and
decoder choices. Subtract effects fixed by the linear readout. Use random
subspaces matched for rank, symmetry, metric, and leading-mode mass.

**Case order:** Begin with a first-branch-dominated case such as the ethane
torsional barrier. Delay difluoroethane gauche preference until the nonlinear
branch is included.

**Success criterion:** A compact mode interaction predicts a held-out change of
geometry or species within a stated scope.

## E10: lean evaluator and architecture transfer

**Question:** Does the discovered function space reduce real computation, and
does any compact basis recur across models?

**Branch evaluator outputs:** Parameter count, storage, FLOPs, wall time, peak
memory, batch-size scaling, and fidelity.

**Cross-model outputs:** Principal angles in a shared function space across
seeds, model sizes, and checkpoints. Compare functions rather than raw channel
indices.

**Architecture follow-up:** Only after stable recurrence is observed, fit a
compact spline or analytic family and test training or fine-tuning with that
parameterization.

**Claim boundary:** Branch-only speed is not full-model speed. A checkpoint
basis that fails to recur can still be a valid post hoc representation.

## First preliminary result package

The smallest credible preliminary package contains five artifacts.

1. A provenance and architecture manifest for MACE-OFF23 small.
2. An exact redundancy table by order and irrep.
3. An untruncated first-branch reconstruction table for energies and forces.
4. A global versus local rank-fidelity figure.
5. Three to six physical mode cards with complete interaction cores.

Items 1 through 3 are exactness results. Item 4 is the first compression result.
Item 5 is the first interpretation result.
