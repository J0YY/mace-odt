# Independent correctness audit: MACE–ODT research package

## Scope and evidence

Reviewed on 2026-09-21:

- `MACE_ODT_Research_Specification.md`, all 27 sections and appendices, extracted from the user's ZIP.
- The complete bundled verification script and manifest.
- Thomas Dooms et al., *Compositionality Unlocks Deep Interpretable Models*, arXiv:2504.02667v1, including Algorithms 1–3 and the diagrams in the ten-page PDF.
- David Olloqui, *Reading the inside of a machine-learned interatomic potential*, the user's complete `shared(2).html`.
- Official MACE `blocks.py` and `models.py`, including the v0.3.6 tag and current main-branch implementation, for interface-level inspection. Neither establishes which implementation was used for the HTML's experiments.

The attached HTML's SHA-256 is `df13c7b44753902606ba6b86296c4d883ece3cdfacf8e752c0a6db7f31882fb8`; this matches the package manifest's `shared(1).html`. The differing filename therefore does not indicate a differing source snapshot. The specification, bundled script, and bundled result hashes also match their manifest entries.

**Conclusion:** The specified first-branch coefficient-space construction has a sound mathematical core. Its application to a real checkpoint, physical usefulness, compression benefits, and mechanistic interpretation are not established by this package. Several source claims need correction or tighter scope. The specification already anticipates many of those problems.

## Files

- `independent_checks.py`: eight additional test groups written independently of the bundled script; NumPy and standard library only.
- `independent_results.json`: actual output of those checks.
- `bundled_checks_original.py`: a copy of the supplied 16-check script, unchanged.
- `bundled_checks_rerun.json`: actual results from rerunning the supplied script.

Run the independent checks with:

```bash
python independent_checks.py --output independent_results.json
```

The independent checks use explicit exceptions rather than Python `assert`, so optimization flags do not disable them. The original script uses `assert` and should be run without `-O`.

**No MACE checkpoint, SAE dictionary, molecular benchmark, training run, or timing experiment was executed.** The angular counts below are representation-theoretic calculations, not tests of a particular checkpoint's serialized Clebsch–Gordan buffers.

## Verified central construction

Let the branch coefficients be K_nu for nu=1,2,3 in an orthonormal primitive function basis, with positive order weights beta_nu. For a shared orthogonal projector P, define

    E(P) = sum_nu beta_nu ||K_nu - P^tensor(nu) K_nu||_F^2
    Gamma = sum_nu beta_nu sum_a rho_(nu,a)
    B(P) = tr((I-P) Gamma).

Here rho_(nu,a) is the partial trace of the outer product of K_nu, retaining slot a.

The commuting-projector inequalities give

    E(P) <= B(P) <= 3 E(P).

A leading eigenspace minimizes B at fixed rank, not generally E. If both the chosen projector and the optimal comparator belong to the same admissible rank/equivariance class, and B is actually minimized over that class, then

    E(P_spectral) <= 3 E(P_optimal).

Thus the norm approximation factor is sqrt(3), as claimed. The 200 extra randomized cases test the sandwich inequality, rather than claiming numerical proof of global optimality.

### The factor is sharp

In two dimensions, take beta_1=beta_3=1,

    K1 = sqrt(3-eta) e1,
    K3 = e2^tensor(3),
    0 < eta < 2,

and retain rank one. Gamma=diag(3-eta,3), so spectral selection keeps e2. Its squared residual is 3-eta, while keeping e1 has squared residual 1. For a general retained unit vector, write t=cos(theta)^2. The retained squared norm is

    (3-eta)t + (1-t)^3.

It is convex on [0,1], and its maximum is the e1 endpoint. Therefore the optimal error really is 1, and the ratio approaches 3 as eta approaches zero. At eta=0.001, the executed test gives a squared-error ratio of 2.999 and norm ratio 1.7317621083740111.

This both confirms the stated guarantee and demonstrates why it must not be relabeled an exact SVD solution of the coupled objective.

## ODT source verification

The specification correctly treats the top-down bond environment as a partial contraction of the full coefficient tensor after upstream orthogonalization. Its interpretation of a Gram eigenvalue as sigma squared is correct.

The original paper's literal squared-Frobenius-Gram tail rule does not imply its stated relative tensor error. The package includes a valid one-layer binary-network counterexample, separately from its generic matrix counterexample. For squared singular values 1 and 0.03 at both relevant bond types, the normalized squared-Gram tail is

    0.03^2 / (1+0.03^2) = 0.00089919072834449,

which passes a 0.1^2/3 threshold, but the relative coefficient-tensor error is

    sqrt(0.03/(1+0.03)) = 0.170664037...

rather than at most 0.1. A tensor Frobenius error uses discarded Gram eigenvalues, not their squares. Likewise, cropping the input legs of an isometric core does not generally preserve that core's row isometry. The specification correctly limits those claims rather than importing them unchanged.

The independent two-layer binary-network check reconstructs an explicit dense tensor after bottom-up QR, and compares recursive intermediate/leaf environments to independently matricized dense tensors. The largest relative error is below 1e-15. This is a synthetic algebra check, not an SVHN reproduction or a deep-MACE compiler.

## Angular quotient verification

For a primitive angular representation V_0e + V_1o + V_2e + V_3o, direct ordered-coupling enumeration and independent weight-multiplicity counting in Sym^3(V) give:

| Output | Ordered paths | Symmetric multiplicity | Nullity |
|---|---:|---:|---:|
| 0e | 23 | 8 | 15 |
| 1o | 51 | 12 | 39 |
| 2e | 65 | 14 | 51 |

This supports the claimed algebraic counts under those representation assumptions. The scalar correlation-order-one and order-two path counts are 1 and 4, respectively; the HTML reverses them in its prose saying four two-body and one three-body coefficient. A 15-dimensional kernel is not necessarily 15 separately zeroable stored-coordinate axes.

## Required specification amendments

1. **Section 12.4, Eq. (30): distinguish the compressed energy.** After nontrivial truncation the left side must be E_L^P, not the original E_L, unless an explicit residual is added or discarded coefficients vanish.
2. **Section 12.1, Eq. (27): cover singular support explicitly.** For rectangular C of full column rank, choose a supported left inverse A with AC=I and write encoder d=A^T z and decoder b=Cz. C^(-T) is undefined for rectangular C. Section 8 already explains much of the necessary support caveat; the export equation should implement it.
3. **Section 12.6 and rank selection: require a cutoff gap or retain an entire degenerate block.** Degeneracy inside a fully retained subspace is harmless to its projector. Degeneracy straddling the cutoff makes the projector nonunique, not merely the basis within it. For K2=diag(1,-1), Gamma=2I; keeping e1 incurs squared error 1, whereas keeping (e1+e2)/sqrt(2) incurs squared error 2. Both choices minimize the marginal bound.
4. **Sections 3.11 and 19/E7: resolve the exact interface and discovery/test chronology.** These are prerequisites to a source replication, not details to infer from a common width of 96.
5. **Section 9.7:** state O(3)-invariance when using parity-separated Schur blocks. Rotation invariance alone cannot in general separate equal-l opposite-parity copies. For the natural-parity primitive basis and full O(3) setting the intended conclusion holds.

## HTML claims that should not be imported uncritically

### Interface, vector, and covector distinction

The identity f_d=d^T F is invariant under F'=MF and d'=M^(-T)d. An additive decoder vector instead transports as v'=Mv. Substituting an SAE decoder into that formula without a dual convention is not valid under general nonorthogonal gauge changes. Nor can a post-product hidden direction automatically be treated as a pre-product radial direction.

The HTML's blanket assertion that a layer-one output basis is restricted to signed permutations/scalings by the next channel-diagonal convolution is not valid at the usual post-product feature interface of the inspected standard MACE code. There are intervening trainable linear consumers. For h=P b, consumers A h, S_z h, and w^T h, the simultaneous changes

    P'=MP,
    A'=A M^(-1),
    S_z'=S_z M^(-1),
    w'^T=w^T M^(-1)

preserve every consumer input. In the inspected implementation these correspond to the product block's final linear, the next interaction's `linear_up`, its `skip_tp`, and the first readout. First-layer residual producers, where present, also require compensation. This does not establish the gauge at a different pre-product interface; the HTML's hook location remains unresolved.

### Invariance is not full functional identifiability

An ordinary channel Gram L transforms as M L M^T, so its Euclidean spectrum is not invariant under general GL transformations. More importantly, invariance under an exhibited family is weaker than being determined by the total input-output function. Appendix B admits that the symmetry family is not exhaustive. Keep all positive claims scoped to a specified interface, tensor target, metric, and covered transformations.

### Benchmark selection and inconsistent summaries

The HTML says candidates were screened for a gain over the whole benchmark and then requalified on a held-out half. If that whole benchmark included the eventual test half, the latter was used for selection and is not an untouched test set. Resolve the chronology or use fresh evaluation paths.

The 12.4–16.3 percent range applies to four SAE directions, not all five: the contrast row gives 3.89 percent. A universal assertion of recovery only at 45 modes also overstates the table: one SAE direction gives 21.97 percent at 20 modes versus 12.38 percent untruncated. Aggregate and per-direction recovery criteria must be distinguished.

### Adaptive optimizer nullspace dynamics

HTML Eq. (22) is not an exact general Adam trajectory. A kernel component of the loss gradient can vanish while the kernel component of a coordinate-preconditioned step is nonzero. For T=[1,2], gradient (1,2), and kernel direction (2,-1), the first normalized Adam step is proportional to (-1,-1), which has nonzero kernel projection. Starting at zero makes the same first-step counterexample valid even when coupled weight decay is enabled. Momentum, state transport, discrete updates, and the particular kernel basis must all be handled. This counterexample does not assert that a particular MACE training run exhibited the effect.

### Feature FVU does not determine readout error

For residual r=h-h_hat, the exact linear-readout mean squared error is

    s^2 w^T E[r r^T] w.

It does not generally equal FVU times s^2 w^T Sigma w. With Sigma=I, w=e1, residual second-moment matrices diag(0.2,0) and diag(0,0.2) have identical FVU=0.1 but readout MSEs 0.2 and 0.0. The HTML's reported 267 meV/site might still be a direct measurement; the prose/equation does not establish that. Request the directly measured residual statistic before using that number as an independently checked result.

Other source qualifications: forces are the negative energy gradient; binary reach-only AUC is not automatically the full magnitude-based AUC; label-table controls contain the labels by construction and do not show that representations contain no chemical information; a finite molecule always has a finite subset expansion, while a nonpolynomial readout lacks a uniform finite body-order ceiling over arbitrary system sizes.

## Empirical gates still open

A real checkpoint implementation must first establish untruncated branch-energy and derivative equivalence, correct angular conventions and quotient ranks, numerical support handling, and all-consumer gauge compensation. Only then should it test rank truncation against matched coefficient/energy/force baselines. Physical derivatives, inference cost, causal interpretation, and the full nonlinear branch require their own evidence. The supplied tests do not establish these results.
