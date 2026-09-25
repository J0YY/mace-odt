# Preliminary real-checkpoint results

## Status

These results establish checkpoint provenance, ordinary forward correctness,
exact repeated-slot path redundancy, an exact first-branch compiler, and the
first global coefficient rank ladders. They do not establish held-out molecular
fidelity or full-model replacement.

## Environment

- Cluster: Athena
- Scheduler: Slurm
- Project directory: `/work/joy/mace-odt`
- Environment: `/work/joy/envs/mace-odt-cpu`
- MACE: 0.3.16
- PyTorch: 2.6.0 CPU
- NumPy: 2.2.6
- e3nn: 0.4.4

The CPU environment is intentional for the initial exactness gates. Later
matrix-free and molecular rank ladders can use a separate GPU environment.

## Checkpoint provenance

- Model: official MACE-OFF23 small
- License reported by the loader: Academic Software License
- File size: 7,347,350 bytes
- SHA-256: `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`
- Audit job: 399950

## Architecture audit

| Quantity | Observed value |
|---|---:|
| Total parameters | 694,320 |
| Interaction blocks | 2 |
| Supported elements | 10 |
| Cutoff | 4.5 angstrom |
| Product output | `96x0e` |
| First readout weights | 96 |
| Nonlinear readout | 96 to 16 to 1 |
| Cubic path-weight shape per layer | `10 x 23 x 96` |

The calculator freezes model parameters for inference, which is why the audit
reports zero parameters with `requires_grad=True`. This does not mean the
serialized model contains no learned parameters.

## Forward correctness smoke tests

| Check | Maximum discrepancy | Tolerance | Result |
|---|---:|---:|---|
| Methane translation | `3.94e-15` | `1e-7` | Pass |
| Methane rotation | `5.21e-15` | `1e-7` | Pass |
| Methane atom order | `2.27e-13` | `1e-7` | Pass |
| Water translation | `3.11e-15` | `1e-7` | Pass |
| Water rotation | `6.69e-15` | `1e-7` | Pass |
| Water atom order | `0` | `1e-7` | Pass |
| Methane force finite difference | `2.05e-8 eV/angstrom` | `2e-4 eV/angstrom` | Pass |

Translation, rotation, and atom-order entries combine energy and force
discrepancies in their respective native units. They are software integration
checks rather than model-accuracy measurements.

## Exact path quotient audit

- Audit job: 399952
- Output irrep in both layers: `96x0e`

| Layer | Correlation order | Coupling shape | Stored paths | Supported paths | Null paths |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | `16 x 1` | 1 | 1 | 0 |
| 0 | 2 | `16 x 16 x 4` | 4 | 4 | 0 |
| 0 | 3 | `16 x 16 x 16 x 23` | 23 | 8 | 15 |
| 1 | 1 | `16 x 1` | 1 | 1 | 0 |
| 1 | 2 | `16 x 16 x 4` | 4 | 4 | 0 |
| 1 | 3 | `16 x 16 x 16 x 23` | 23 | 8 | 15 |

For the cubic map, the numerical null residual is `3.27e-15` and the support
plus null projector residual is `3.10e-15`. A deterministic random functional
test produced the following norms.

- Null direction response: `2.50e-16`
- Retained direction response: `4.73`
- Relative null response: `5.28e-17`

This matches the supplied HTML's scalar count of 23 stored paths and eight
independent repeated-slot combinations. The identical count in both layers is
an architectural fact about their shared coupling construction.

## Live quotient integration

- Integration job: 399969
- Random feature maximum error: `2.84e-13`
- Both product layers replaced, maximum energy error: `0 eV`
- Both product layers replaced, maximum force error: `8.88e-16 eV/angstrom`

The replacement stores only the supported path coordinates. It does not reduce
the 96-channel representation and is not a learned low-rank result.

## Registered radial function interface

- Interface job: 399973
- Structural measure: equal neighbor-species weight
- Radial measure: normalized uniform distance from 0.5 to 4.5 angstrom
- Quadrature: 128-point Gauss-Legendre
- Angular measure: uniform sphere with component-normalized harmonics

All four angular blocks and all ten central species have numerical support rank
96 under the strict floating support threshold. The smallest retained singular
values range from about `9.65e-10` to `5.49e-7`. The upstream function map is
therefore full rank but very ill-conditioned.

The relative metric change between 64-point and 128-point quadrature is at
most `9.12e-7`, below the declared convergence tolerance of `2e-6`.

On methane and water, the reconstructed aggregate density differs from the
native interaction by at most `3.33e-16`. First-branch energies differ by at
most `1.78e-15 eV`, and forces differ by at most `1.83e-15 eV/angstrom`.

## Exact first-branch compiler

- Compiler job: 399988
- Random aggregate-feature maximum error: `6.54e-13 eV`
- Repeated-slot symmetry residual through order three: `3.55e-15`
- Cubic path-null coefficient change: `4.44e-15`
- Methane and water branch energy maximum error: `8.88e-16 eV`
- Methane and water branch force maximum error: `9.99e-16 eV/angstrom`

The compiled tensors include the learned path coefficients, symmetric angular
couplings, post-product linear map, first scalar readout, and checkpoint scale.
They do not include the second interaction branch or atomic reference energies.

## Global mixed-order environment

- Equal-order job: 399993
- Coefficient-balanced job: 399994

The implementation contracts every slot separately, including all channel and
path cross terms. It then absorbs the registered upstream radial factors and
extracts complete O(3) multiplicity blocks. Across all central species:

| Diagnostic | Maximum observed value |
|---|---:|
| Relative off-block norm | `8.11e-19` |
| Relative magnetic-copy deviation | `4.44e-17` |
| Relative trace identity residual | `1.32e-15` |

Every tested rank satisfies the actual tied-error bound and its factor-three
reverse comparison within the declared numerical tolerance.

Equal order weights make the linear term dominant. The coefficient-balanced
arm uses `beta[z, order] = 1 / ||K[z, order]||^2`, so every order contributes
equally to the direct-sum coefficient norm before occurrence counting.

| Uniform multiplicity rank per irrep | Best species error | Worst species error |
|---:|---:|---:|
| 4 | `0.0508` | `0.211` |
| 8 | `0.0163` | `0.0818` |
| 16 | `0.00326` | `0.0196` |
| 32 | `0.000243` | `0.00125` |
| 48 | `7.28e-6` | `6.74e-5` |
| 64 | `8.43e-8` | `1.14e-6` |

These entries are relative squared coefficient errors. They are not energy or
force errors on a molecular test set.

## End-to-end projector diagnostic

- Projector job: 399999
- Projector form: native encoder followed by native decoder
- Geometries: methane and water only

The sequential map is necessary because the full upstream radial factor is
ill-conditioned. Forming `C Pi C+` as one dense native matrix caused a failed
random-baseline algebra diagnostic. Applying the encoder and decoder separately
reduced the relative idempotence residual to at most `4.93e-9` for truncated
random maps and about `1.5e-14` for the global maps.

| Rank per irrep | Global max energy error | Local radial max energy error | Global max force error | Local radial max force error |
|---:|---:|---:|---:|---:|
| 16 | `0.112 eV` | `0.616 eV` | `0.172 eV/angstrom` | `0.448 eV/angstrom` |
| 32 | `0.0342 eV` | `0.180 eV` | `0.0330 eV/angstrom` | `0.162 eV/angstrom` |
| 48 | `0.0131 eV` | `0.0374 eV` | `0.00600 eV/angstrom` | `0.0154 eV/angstrom` |
| 64 | `0.00138 eV` | `0.00556 eV` | `0.00168 eV/angstrom` | `0.00855 eV/angstrom` |

At full rank, the evaluator now applies every encoder and decoder rather than
taking an identity shortcut. The worst energy or force discrepancy is
`1.71e-10`, and the global and local bases agree at about `6.22e-15` or better.
This passes the declared `5e-10` integration gate. The lower-rank comparison is
encouraging but uses only two integration geometries. It is not a held-out
benchmark or an uncertainty estimate.

## Frozen held-out branch fidelity

- Dataset: official MACE-OFF23 test archive
- Repository DOI: `10.17863/CAM.107498`
- Full test set: 50,195 configurations
- Frozen metadata-stratified subset: 64 configurations
- Fidelity job: 400007
- Random controls: three fixed canonical seeds

The subspaces were derived only from checkpoint weights and the declared
function measure. The evaluation configurations were selected before projected
errors were computed.

| Rank per irrep | Global mean energy error per atom | Local radial mean energy error per atom | Global mean force RMSE | Local radial mean force RMSE |
|---:|---:|---:|---:|---:|
| 16 | `0.0312 eV` | `0.144 eV` | `0.193 eV/angstrom` | `0.278 eV/angstrom` |
| 32 | `0.00312 eV` | `0.0188 eV` | `0.0573 eV/angstrom` | `0.126 eV/angstrom` |
| 48 | `0.00158 eV` | `0.00311 eV` | `0.0249 eV/angstrom` | `0.0429 eV/angstrom` |
| 64 | `0.000248 eV` | `0.000594 eV` | `0.00319 eV/angstrom` | `0.00921 eV/angstrom` |
| 80 | `0.0000249 eV` | `0.000124 eV` | `0.000306 eV/angstrom` | `0.00120 eV/angstrom` |

For local minus global mean force RMSE, every paired 95 percent bootstrap
interval excludes zero. At rank 64 the interval is `0.00533` to `0.00660
eV/angstrom`. At rank 80 it is `0.000832` to `0.000961 eV/angstrom`.

This is the first evidence that downstream-composed selection provides a real
fidelity advantage over local radial SVD at matched function rank. It remains a
single-checkpoint, first-branch result. The next test must insert the maps at the
shared interface and measure every downstream consumer.

## Shared-interface boundary

- Shared-interface jobs: 400013 and 400023
- Insertion point: first interaction density before product block zero
- Consumers affected: first scalar readout and the later nonlinear branch

| Rank per irrep | Global shared mean force RMSE | Local radial shared mean force RMSE | Global shared mean energy error per atom | Local radial shared mean energy error per atom |
|---:|---:|---:|---:|---:|
| 48 | `0.0327 eV/angstrom` | `0.0316 eV/angstrom` | `0.00406 eV` | `0.00193 eV` |
| 64 | `0.00925 eV/angstrom` | `0.00763 eV/angstrom` | `0.000754 eV` | `0.000420 eV` |
| 80 | `0.00174 eV/angstrom` | `0.000913 eV/angstrom` | `0.0000735 eV` | `0.0000986 eV` |

The first-branch global basis is therefore not a general full-model basis. The
later nonlinear consumer changes which directions matter. Generalizing the ODT
goal now means constructing a multi-consumer environment, not claiming that the
first-readout environment already compresses all of MACE.

For shared force RMSE at rank 64, the paired local-minus-global 95 percent
interval is `-0.00217` to `-0.00106 eV/angstrom`. At rank 80 it is `-0.000992`
to `-0.000686 eV/angstrom`. Negative values mean local radial SVD is better for
the full shared-interface target at these ranks.

## Data-assisted two-consumer pilot

- Discovery manifest job: 400135
- Gradient environment job: 400136
- Frozen fidelity job: 400155
- Discovery set: 128 label-free configurations
- Evaluation set: the existing frozen 64 configurations
- Interface: first interaction density before product block zero

The primary basis averages gradient Grams from the first linear readout and the
later nonlinear readout after giving both consumers equal global trace. A second
basis uses the gradient of their summed energy and therefore preserves branch
cancellation. These are additive local sensitivity bases. They are not exact
ODT environments or finite projector optima.

The T1 graph checks close at `4.44e-16`, branch-gradient addition closes at
`2.07e-15`, and every stored radial curve matches the live checkpoint exactly.
The largest radial metric reconstruction residual is `1.34e-14`. The full-rank
held-out evaluator has a worst total-energy or force error of `2.41e-9`, below
the `5e-9` gate.

| Rank per irrep | Balanced gradient energy error per atom | Local energy error per atom | Balanced gradient force RMSE | Local force RMSE |
|---:|---:|---:|---:|---:|
| 48 | `0.000594 eV` | `0.00193 eV` | `0.0226 eV/angstrom` | `0.0316 eV/angstrom` |
| 64 | `0.000127 eV` | `0.000420 eV` | `0.00495 eV/angstrom` | `0.00763 eV/angstrom` |
| 80 | `0.0000171 eV` | `0.0000986 eV` | `0.000499 eV/angstrom` | `0.000913 eV/angstrom` |

At rank 64, local minus balanced-gradient force RMSE is `0.00268
eV/angstrom`, with paired 95 percent interval `0.00226` to `0.00308`. The
Holm-adjusted sign-flip probability is `0.000200`. The energy advantage is
`0.000293 eV` per atom, with interval `0.000217` to `0.000372`.

The weak feasibility rule passes. The strong rank-64 rule fails because force
RMSE remains above `0.001 eV/angstrom` and energy error remains above `0.0001
eV` per atom. Rank 80 passes both absolute point thresholds, but retains five
sixths of each 96-dimensional multiplicity space. This is evidence for useful
downstream-aware selection, not yet evidence for a very lean compiled model.

The summed-energy-gradient basis is better for force fidelity than the balanced
basis at ranks 64 and 80. At rank 64 its force RMSE is `0.00404
eV/angstrom`. Its separate linear and nonlinear branch errors are much larger,
which shows that branch cancellation is real. The balanced basis is therefore
the safer analysis representation. The summed basis is currently the better
output-fidelity representation.

The consumer-specific ablation supports the intended attribution. At rank 64,
the nonlinear-only gradient basis improves force RMSE over the linear-only basis
from `0.00960` to `0.00801 eV/angstrom`. It reduces nonlinear-branch error from
`0.000364` to `0.0000794 eV` per atom and head-preactivation RMSE from `0.0117`
to `0.00476`. Every paired interval for these differences excludes zero, with
sign-flip probability `0.000100`. Combining both consumers is still materially
better than either consumer alone.

Activation PCA is much worse, with rank-64 force RMSE `0.177
eV/angstrom`. Five canonical random controls range from `7.01` to `27.5
eV/angstrom` at rank 64. The improvement is tied to downstream sensitivity and
is not generic variance preservation.

Rare-species blocks remain underdetermined. Phosphorus appears in four discovery
configurations, bromine in eight, and iodine in two. One iodine configuration
appears in each deterministic fold. Iodine fold comparison establishes only
that the calculation is defined. It does not establish a stable iodine mode.

The exact next test is a coefficient environment for the first readout, the
second-interaction message density, and its skip path. That construction remains
degree three at product block zero and can recover a Dooms-aligned marginal
tail bound for its declared immediate-consumer norm.

## Result files

| File | SHA-256 |
|---|---|
| `results/athena_environment.json` | `a41521bec527bea6f902c6ab0451c0a3da74cb950c9ac99add73c9b87a9522b6` |
| `results/athena_environment_job.json` | `16a58630ced6980d3400063433709c8a8e730d2076a780789a42cad61744b8bf` |
| `results/mace_off23_small_checkpoint_audit.json` | `a6af05bfa8c487738e23fa3ee83a88417a4f3787e4e04a32e1434df1edccfcc2` |
| `results/mace_off23_small_path_quotient.json` | `022f32d297a0691c650f7d02dc20236e48faa270b8c01ada557ddb56df71f18f` |
| `results/mace_off23_small_quotient_integration.json` | `582789a202f6f83703016e658dcee4630211a100c016b31afa26be6c35873997` |
| `results/mace_off23_small_radial_interface.json` | `675f6c7771b7c9b22321b0c69c94adbd6b825e98cd195ea0d616d2f2c5960a53` |
| `results/mace_off23_small_first_branch.json` | `c9a3d916f5823a0a3aedf8233ff27a0b4dcc9a674e8f775dcbb9760cc99999cc` |
| `results/mace_off23_small_global_environment.json` | `a3975acc08525ea8aef19abe1e9a78801d14ebbd33bac682f53162ea3ed5c3e6` |
| `results/mace_off23_small_global_environment_balanced.json` | `082deae6e5f1d1416c15d0c28c201348ed208a7773b93965bee89886c86267f8` |
| `results/mace_off23_small_projected_branch_balanced.json` | `7980ea21baea63c25f11f7627df00d2851adf9682ad13a9c3da242b623fa2f14` |
| `results/mace_off23_test_manifest_64.json` | `5e7314c41fedbb614a457d7f4b91c87e615f08f95a35e843b224cd0b9aa82835` |
| `results/mace_off23_small_heldout_branch_balanced_64.json` | `80290e13ac4c68c7b5d7469b7f538b355162c9b0bcf5da8646d1160dce0fe298` |
| `results/mace_off23_small_shared_interface_balanced_64.json` | `08763df73a35e4e6171c6388c1f82217644a9522023b661a6a4ef362c9ca7903` |
| `results/mace_off23_discovery_manifest_128.json` | `a624def23b363155da28a148998802438ab756f7c698161bc388ebed230d653c` |
| `results/mace_off23_small_multi_consumer_environment_128.json` | `26c1ca405ccfa09b66438c301548dd686e36ca897c6fd62450b27d9fab89f46f` |
| `results/mace_off23_small_multi_consumer_fidelity_64.json` | `8e0741c33dd64066e160b33703e956c3306ec9f2282c45bf28325ba9b56db32c` |
| `results/mace_off23_small_immediate_consumer_environment.json` | `f3037883b3baa116d8512aa88e340f7fea040561e1d0000fa90cbe0dea76cc51` |
| `results/mace_off23_small_immediate_consumer_fidelity_64.json` | `e871b41b41caf089b1a98acad881ad14007d9e1d4aca7304bec99ed50d16d86f` |
| `results/mace_off23_small_immediate_consumer_fidelity_64.provenance.json` | `2039abbf7dc73043b45d02a5f34260ede71fc9842ce4df48917bafdc8d6fb555` |

## Immediate next implementation

1. Compile vector-valued product-zero coefficients for the first readout,
   second-interaction message density, and second-interaction skip path.
2. Construct the exact immediate-consumer marginal operator with all path cross
   terms and a declared output metric.
3. Validate its matrix-free action against a dense tiny-graph oracle and test the
   tied factor-three bound.
4. Evaluate its basis through the same frozen T2 rank ladder.
5. Only after the exact method passes, compile the sixteen nonlinear-head
   preactivations as a factorized degree-nine target.

## Exact immediate-consumer environment, September 24

The exact coefficient environment now compiles for the frozen MACE-OFF23 small
checkpoint. It includes the first scaled readout, the second interaction
message density, and the second interaction skip path. The message term uses a
declared one-free-directed-edge measure with the frozen 128-point radial
quadrature and uniform checkpoint species weights. It uses no DFT labels and no
discovery geometry activations.

All exactness gates passed. The largest independent numerical residual was
`1.06e-14`, against a tolerance of `5e-9`. The analytic product coefficients
reconstruct the native product, the complete-irrep message pullback is rotation
invariant, the environment trace identity holds, and the tied-projector loss
obeys the degree-three factor bound under the declared metric.

The spectrum is sharply concentrated. For the preregistered globally
trace-balanced consumer objective, the retained trace fractions are:

| Multiplicity rank per irrep | Retained trace fraction |
|---:|---:|
| 16 | 0.997706 |
| 32 | 0.999868 |
| 48 | 0.999992 |
| 64 | 0.999999805 |
| 80 | 0.999999998 |
| 96 | 1.0 |

This is encouraging evidence that the exact declared objective has a compact
functional spectrum. It is not yet evidence that finite simultaneous
projection preserves the full potential. The frozen 64-structure held-out
rank ladder is the deciding test. It evaluates energy, forces, all three
immediate consumers, and the final nonlinear output without retraining.

## Held-out immediate-consumer result, September 24

The frozen T4 run completed on all 64 held-out structures and all 84 declared
method-rank combinations. The rank-96 replay gate passed. Its largest stored
energy or force discrepancy was `2.41e-9`, below the frozen `5e-9` tolerance.

The preregistered trace-balanced exact method did not pass the weak or strong
rank-64 rule. It also triggered the preregistered no-go rule for this objective
and measure.

| Method | Rank | Force RMSE, eV/A | Energy error, eV/atom |
|---|---:|---:|---:|
| Exact immediate consumers, trace balanced | 64 | 0.008060 | 0.000568 |
| Local radial SVD | 64 | 0.007631 | 0.000420 |
| Prior exact first-branch basis | 64 | 0.009254 | 0.000754 |
| Total-energy-gradient basis | 64 | 0.004040 | 0.000151 |
| Exact immediate consumers, trace balanced | 80 | 0.000986 | 0.0000429 |
| Local radial SVD | 80 | 0.000913 | 0.0000986 |
| Total-energy-gradient basis | 80 | 0.000446 | 0.0000146 |

At rank 64, the exact method significantly improved on the prior exact
first-branch basis. The paired force improvement was `0.001194 eV/A`, with a
95 percent interval from `0.000580` to `0.001818` and Holm-adjusted
`p = 0.0010`. This confirms that including the actual downstream message and
skip consumers improves the earlier exact ODT construction.

It did not beat local radial SVD or the total-energy-gradient basis. The paired
force difference against local radial SVD was negative and its interval crossed
zero. The total-energy-gradient basis was decisively better at ranks 64 and 80.
At rank 80, the exact method met the absolute force and energy targets, but it
still lost both preregistered relative comparisons. It therefore cannot be
claimed as a superior compact direction-selection rule.

The result rejects the present exact immediate-consumer objective under its
frozen discrete measure. It does not reject the broader goal of a lean
physically motivated MACE parameterization. The sharp coefficient spectrum was
not enough to predict held-out force fidelity. Small omitted directions can be
amplified by simultaneous intervention at every node and by the remaining
nonlinear graph. A successful analytic objective will need a closer proxy for
whole-model force sensitivity, or a hybrid with the total-energy-gradient
basis, while preserving the exact equivariant block structure.
