# Graph-pullback ODT objective and next tests

## Decision after T4

T4 rejects the frozen immediate-consumer coefficient objective as a competitive
rank-64 or rank-80 full-model compressor. It does not reject functional
compression, downstream-aware selection, or a nonlinear extension of ODT.

The current exact operator solves a declared local coefficient problem. The
desired quantity is the energy and force error caused by inserting one tied
projector at every applicable node of the complete frozen graph. No proved
constant connects those two quantities. T4 shows that this missing link matters
in practice.

## Relation to Dooms et al.

The proposed pivot is farther from the literal Algorithms 1 through 3 in Dooms
et al. Their construction is weight-only, uses true tensor-network separators,
and selects a bond subspace by one eigendecomposition. The nonlinear MACE cut is
shared across nodes and is followed by message aggregation, another product,
and a nonlinear readout. Its finite intervention loss is input-dependent and
nonquadratic in the tied projector.

The pivot remains close to the central ODT principle. The retained space is
still chosen by pulling the rest of the computation back to a registered
internal cut. It still uses the exact functional metric, complete irreducible
multiplets, and one shared projector. What changes is the downstream metric and
the optimizer.

The project will distinguish three names.

1. **Structural ODT** is the existing exact coefficient environment and its
   factor-three coefficient-norm result.
2. **Graph-pullback ODT** uses full-model energy and force sensitivity at the
   same functional cut. It is an ODT extension, not the original algorithm.
3. **ODT-parameterized frozen compression** directly optimizes finite projected
   energy and force fidelity. It uses the ODT representation but is not a
   closed-form ODT decomposition.

No graph-pullback or finite-loss result will inherit the current coefficient
tail theorem automatically.

## Correct nonlinear target

Let $h_X(R)$ be the canonical first-density representation for configuration
$X$. For species and irrep block $b=(z,\ell,p)$, let

\[
P_b=U_bU_b^\top,
\qquad
Q_b=I-P_b.
\]

This is the canonical functional projector. If the native density is
$\rho_b=F_bh_b$ and $F_b^\dagger$ is the registered supported left inverse,
the native action is

\[
\rho_b\longmapsto F_bP_bF_b^\dagger\rho_b.
\]

Equivalently, the canonical gradient is
$g_{h_b}=F_b^\top g_{\rho_b}$. Applying $P_b$ directly to native channels is
not valid. A projector may mix multiplicities only within equivalent
$(\ell,p)$ copies and is tensored with the identity on all $2\ell+1$ magnetic
components. The current cut has one registered parity for each $\ell$. The same
projector is used at every node of species $z$. This preserves permutation
symmetry and $O(3)$ equivariance exactly.

The direct finite target is

\[
\mathcal L_{\mathrm{finite}}(P)
=
\mathbb E_X\left[
\frac{|E_P(X)-E(X)|^2}{N_X^2\epsilon_E^2}
+
\frac{\|F_P(X)-F(X)\|^2}{3N_X\epsilon_F^2}
\right].
\]

Here $E$ and $F$ are outputs of the same frozen checkpoint. No DFT labels
are used and no MACE parameter is updated.

For a rigorous nonlinear path, set $Q=I-P$,

\[
h_t=(I-tQ)h_X,
\qquad
E_t=D(R,h_t),
\qquad
s_t=\langle \nabla_hD(R,h_t),Qh_X\rangle.
\]

Write $F_t=-d_RE_t$, where $d_R$ is the total coordinate derivative through the
cut representation and every explicit coordinate dependence in the suffix.
The symbol $\nabla_hD$ is the partial derivative with respect to the registered
cut input. The projector is fixed and independent of $R$.

Within one fixed-neighbor-graph cell and under the smoothness required to
interchange the path and coordinate derivatives,

\[
E_1-E_0=-\int_0^1s_t\,dt,
\qquad
F_1-F_0=\int_0^1d_Rs_t\,dt.
\]

Jensen's inequality gives the per-configuration bound

\[
\frac{|E_1-E_0|^2}{N_X^2\epsilon_E^2}
+
\frac{\|F_1-F_0\|^2}{3N_X\epsilon_F^2}
\leq
\int_0^1\left[
\frac{s_t^2}{N_X^2\epsilon_E^2}
+
\frac{\|d_Rs_t\|^2}{3N_X\epsilon_F^2}
\right]dt.
\]

Here $d_Rs_t$ is the total coordinate derivative through both $h_X(R)$ and
$h_t(R)$. This path objective includes the nonlinear suffix, simultaneous edits
at all nodes, branch cancellation, and coordinate derivatives. It is generally
nonconvex in the tied projectors and has no one-shot eigendecomposition.

## First implementable objective

The first implementation will use the exact tangent at $t=0$,

\[
\mathcal L_{\mathrm{tan}}(P)
=
\mathbb E_X\left[
\frac{s_0^2}{N_X^2\epsilon_E^2}
+
\frac{\|d_Rs_0\|^2}{3N_X\epsilon_F^2}
\right].
\]

This is the squared norm of the exact infinitesimal energy and force derivative
along the declared interpolation. It is not the finite intervention loss and is
not by itself a bound on that loss. It includes cross-node, cross-path, and
cross-block cancellation. It must be optimized jointly over the product of
species and irrep Grassmann manifolds. Replacing the squared sums with
independent block traces is a different upper-bound surrogate and must be
labeled as such.

The structural environment can remain as a regularizer,

\[
\mathcal L_{\mathrm{hybrid}}(P)
=
\mathcal L_{\mathrm{tan}}(P)
+
\eta\sum_b(2\ell_b+1)
\operatorname{Tr}[(I-P_b)\Gamma_b^{\mathrm{struct}}].
\]

The sensitivity term targets the first-order desired behavior. The structural
term favors subspaces that retain the declared coefficient organization. It
does not replace the sensitivity term.

## Spectral global-adjoint approximation

D1 must be derived from the same energy and force tangent metric as D2. It is
not the existing total-energy-gradient Gram.

Write the coherent tangent scalar as

\[
s_0=\sum_b s_b,
\qquad
s_b=\operatorname{Tr}(Q_bS_b),
\]

where $S_b$ contains the activation-gradient contraction summed over all nodes
and magnetic components in block $b$. Define

\[
A_b=\operatorname{sym}(S_b),
\qquad
B_{b,a}=\operatorname{sym}(d_{R_a}S_b).
\]

Only these symmetric parts contribute. Because $Q_b$ is fixed with respect to
coordinates,

\[
d_{R_a}s_b=\operatorname{Tr}(Q_bT_{b,a}),
\qquad
T_{b,a}=B_{b,a}.
\]

For discarded rank $k_b=\operatorname{rank}(Q_b)$, Frobenius Cauchy-Schwarz
gives

\[
\operatorname{Tr}(Q_bS_b)^2
\leq
k_b\operatorname{Tr}(Q_bA_bA_b^\top),
\]

with the analogous bound for every $B_{b,a}$. Let $m_X$ be the number of
species and irrep blocks present in configuration $X$. The frozen unweighted
block inequality is

\[
\left(\sum_{b\in X}x_b\right)^2
\leq
m_X\sum_{b\in X}x_b^2.
\]

Absent species blocks are omitted and do not contribute to $m_X$. Combining
the two inequalities produces the rank-dependent spectral operators

\[
\Gamma_b^{\mathrm{D1-E}}
=
\mathbb E_X\left[
\mathbf 1[b\in X]\frac{m_Xk_bA_bA_b^\top}
{N_X^2\epsilon_E^2}
\right]
\]

and

\[
\Gamma_b^{\mathrm{D1-EF}}
=
\Gamma_b^{\mathrm{D1-E}}
+
\mathbb E_X\left[
\mathbf 1[b\in X]\frac{m_Xk_b\sum_aB_{b,a}B_{b,a}^\top}
{3N_X\epsilon_F^2}
\right].
\]

The expectation uses the same uniform configuration weights as the finite
target. For variable-rank allocation, $k_b$ is recomputed for every proposed
allocation before its spectra are compared. D1-E isolates the effect of an
activation-dependent global energy pullback. D1-EF then adds force sensitivity.
Leading eigenvectors minimize the resulting additive upper bound at fixed block
ranks. Both D1 variants preserve coherent node effects inside each $A_b$, but
drop cross-block cancellation and introduce the within-block rank bound and the
block-sum bound. Random probes may estimate the D1-EF force-output sum. These
approximations and their constants must be reported.

The existing total-energy-gradient basis is a separate baseline called G0. It
uses a geometry-conditioned output gradient Gram but omits the activation
factor in $S_b$ and omits force sensitivity. It is not D1-E.

## Dooms-proximity ablation

The next study must compare increasingly nonclassical objectives rather than
silently changing the meaning of ODT.

| Level | Objective | Relation to Dooms | Terms omitted |
|---|---|---|---|
| D0 | Existing structural coefficient environment | Closest implemented exact specialization | Full nonlinear pullback and forces |
| G0 | Existing total-energy-gradient Gram | Geometry-conditioned active-subspace baseline | Activation dependence and force sensitivity |
| D1-E | Energy global-adjoint upper bound followed by eigendecomposition | Keeps top-down pullback and spectral truncation | Force sensitivity, coherent block sum, and finite remainder |
| D1-EF | Energy-force global-adjoint upper bound followed by eigendecomposition | Closest force-aware spectral extension | Coherent block sum, bound tightness, joint optimization, and finite nonlinear remainder |
| D2 | Joint full-graph tangent objective on Grassmann manifolds | Keeps the cut and pullback principle | Finite nonlinear remainder |
| D3 | Direct finite frozen-model loss | Uses the ODT functional parameterization | No closed-form ODT theorem |

The minimum-deviation conclusion will use equivalence tests rather than visual
similarity. The earliest level that beats G0 and local radial SVD and is
noninferior to every later level is the supported extension. The paired
noninferiority margins are $10^{-4}$ eV per angstrom for mean force RMSE and
$10^{-5}$ eV per atom for mean energy error. A full minimum-departure claim
requires D1-E, D1-EF, D2, and D3 to complete. If some rung is infeasible, the
claim is only the earliest supported level among completed rungs. If D1-EF is
noninferior to D2 and D3, the final method can remain close to the Dooms
spectral algorithm. If D2 is required, the separable spectral upper bound is
insufficient. This result alone does not identify whether block coherence,
bound looseness, or joint optimization is the cause. If only D3 works, the
correct result is a structured post-training compression method, not an ODT
generalization theorem.

## Experimental sequence

### T5, existing-artifact diagnosis

This stage requires no new checkpoint evaluations.

- Compute principal angles and cross-objective trace capture for structural,
  local, total-energy-gradient, and single-consumer bases.
- Identify the discarded directions that distinguish the structural and
  total-energy-gradient spaces.
- Simulate nonuniform species and irrep rank allocation under fixed total
  coordinate budgets.
- Form trace-normalized structural plus total-gradient hybrid matrices for a
  frozen coefficient ladder.

This stage is diagnostic. It cannot make a new held-out compression claim
because the existing 64 configurations have already been inspected.

### T6, tangent calibration

- Use discovery configurations only and cross-fit every calibration. Construct
  or tune on fold A and measure tangent-versus-finite prediction on fold B,
  then swap the folds. Freeze the final T6 rule before refitting on all 128
  discovery structures.
- Compute energy tangent loss exactly for every existing basis at ranks 48, 64,
  and 80.
- Compute the D2 force tangent exactly as the total derivative of the scalar
  $s_0$. This requires one reverse-mode coordinate derivative and no force
  component probes.
- For D1-EF only, check 8-probe and 16-probe convergence for the force-output
  operator against exact component enumeration on a small molecular subset.
- Compare tangent rankings with exact finite projected losses on discovery
  folds.
- Compare G0, D1-E, D1-EF, and the joint D2 objective to isolate activation
  dependence, force sensitivity, and coherent tied terms.

The tangent construction advances only if both cross-fit directions achieve
Spearman rank correlation at least 0.8 across methods separately at each of
ranks 48, 64, and 80. The ranked quantity is the normalized combined finite
energy and force target defined above. The primary correlation includes only
the frozen named nonrandom methods: D0, G0, local radial SVD, first-branch ODT,
immediate-consumer ODT, D1-E, and D1-EF when available. Random and canonical
reverse controls are reported separately and cannot satisfy the gate. At each
rank the tangent correlation must improve by at least 0.2 over the D0
coefficient-tail ranking. The T4 D0-versus-G0 ordering is reported as an
external-consistency diagnostic and is not a gate.

Freeze the D1-EF small-molecule subset before inspecting any probe result. For
blocks whose exact Frobenius norm is at least $10^{-8}$ times the largest block
norm, the 16-probe estimate must have at most 5 percent aggregate relative
Frobenius error when block errors are weighted by exact operator trace, and at
most 10 percent error in every block. Smaller
blocks must have absolute Frobenius error at most $10^{-9}$ times the largest
exact block norm. Probe seeds and normalization must be fixed in the T6
configuration.

Before calibration, require four numerical gates.

1. $Q=0$ gives zero tangent loss and full-rank finite replay error below the
   existing float64 threshold.
2. Finite differences satisfy
   $(E_t-E_0)/t\to-s_0$ and
   $(F_t-F_0)/t\to d_Rs_0$ on several molecules and blocks.
3. Rotation preserves the scalar tangent losses and rotates the force tangent.
4. The coherent $s_0$ equals the sum of separately accumulated node and block
   contractions before squaring.

### T7, projector optimization

- Optimize rank-64 and rank-80 projectors separately on a product of
  Grassmann manifolds.
- Initialize from structural ODT, total-energy-gradient, local radial SVD, and
  deterministic random subspaces.
- Keep the MACE checkpoint frozen.
- Use $\epsilon_E=10^{-4}$ eV per atom and $\epsilon_F=10^{-3}$ eV per
  angstrom unless a new protocol is frozen before evaluation.
- Use deterministic nested discovery folds to select the structural
  regularization coefficient and optimizer settings. When outer fold A is the
  construction half, split A into two fixed inner folds for selection and
  leave outer fold B untouched for scoring. Then swap A and B. Include
  $\eta=0$ and compare D2 and D3 under matched step, restart, and initialization
  budgets.
- Require stable held-fold loss and a small Riemannian gradient or KKT residual
  before freezing a basis. Projector distance is only a diagnostic in blocks
  with a declared spectral gap because degenerate tails can rotate without
  changing the represented space or loss.

Before a full optimization run, use four to eight small structures to verify
the total derivative $d_Rs_0$, gradients with respect to every projector, and
the D3 force-loss gradient against finite differences or an exact tiny-system
oracle. Freeze the batching, checkpointing, maximum steps, restart count,
convergence tolerance, memory limit, and wall-time limit after this preflight.

After both outer directions finish, choose the single hyperparameter setting
with the lowest pooled out-of-fold normalized loss. Resolve exact ties by
smaller $\eta$, then lower compute budget, then lexicographic configuration
hash. Refit that setting once on all 128 discovery structures with the frozen
restart seeds. Among converged restarts, retain the smallest all-discovery
objective, with the smallest seed as the final tie break. This deterministic
collapse rule is frozen before any T8 structure is loaded.

Uniform per-block rank is the primary allocation for direct continuity with
T4. A secondary variable-rank study must match the total number of retained
$O(3)$-expanded canonical coordinates, not only the number of multiplicity
channels. It must also report unexpanded multiplicity count, projector
parameter count, runtime, and memory. Its allocation rule is selected on
discovery folds and then frozen.

D2 and D3 require a torch-native differentiable projector injection. The
existing NumPy-backed projected product wrapper stores detached maps and cannot
optimize $U_b$. D3 must construct forces with higher-order autograd enabled so
gradients reach the projectors. Every MACE and e3nn operation on that route must
pass a grad-grad gate.

Finite projector evaluation must bypass ASE result caching or clear the
calculator results whenever only the internal projector changes. Otherwise ASE
can return an energy or force computed under a previous projector.

### T8, fresh held-out decision

Before evaluating any candidate on T8 structures, freeze and hash the new
evaluation manifest rule, random seed, dataset hash, checkpoint hash, and all
exclusion hashes. The manifest must be disjoint from the T0 discovery set and the original
64-configuration evaluation set by source index, exact ordered structure hash,
and a species-plus-pair-distance invariant fingerprint. Rigid motions and atom
permutations of an inspected structure are therefore excluded.

The invariant fingerprint is defined for the nonperiodic molecular data used
here. It contains the sorted atomic-number multiset and the sorted multiset of
tuples $(\min(Z_i,Z_j),\max(Z_i,Z_j),q_{ij})$, where $q_{ij}$ is the pair
distance rounded to $10^{-6}$ angstrom. Hash matches are excluded
conservatively without collision resolution. Periodic structures are
ineligible unless a separate cell-aware canonical fingerprint is frozen before
manifest generation.

The primary candidate is selected using pooled out-of-fold discovery
predictions only. Choose the earliest of D1-E, D1-EF, D2, and D3 that beats G0
and local radial SVD and is noninferior to every later completed level. If all
four rungs complete, this identifies the minimum supported departure. If a rung
is infeasible, it identifies only the earliest supported level among completed
rungs. Rank 64 is primary if it passes the frozen discovery gates. Rank 80 is
the declared fallback. Commit the chosen method, rank, and basis hashes before
loading the T8 structures. No secondary cell can replace or rescue the primary
result after T8 is opened.

Discovery-stage superiority uses paired baseline-minus-candidate differences.
Discovery-stage noninferiority uses paired candidate-minus-comparator
differences. Require one-sided 95 percent bootstrap bounds above zero for force
superiority, at or below $10^{-4}$ eV per angstrom for force noninferiority, and
at or below $10^{-5}$ eV per atom for energy noninferiority. Also compute paired
one-sided sign-flip $p$ values. For a noninferiority test, subtract the declared
margin from each candidate-minus-comparator difference before testing whether
its mean is below zero. Apply Holm correction to the planned sign-flip tests
separately within the force and energy families. Both the confidence-bound and
Holm-adjusted $p<0.05$ conditions must pass.

Use one evaluator and one full-rank replay gate for every method. Report D0,
G0, D1-E, D1-EF, D2, D3 when feasible, local radial SVD, the previous
first-branch basis, and matched random controls as secondary cells at the same
ranks.

For each configuration, orient paired differences as baseline minus candidate.
The primary force claim requires a positive 95 percent paired confidence
interval and a Holm-adjusted paired sign-flip $p<0.05$ against both G0 and local
radial SVD. For energy, the one-sided 95 percent upper confidence bound on each
paired candidate-minus-baseline difference must be at or below the frozen
$10^{-5}$ eV per atom margin. For each energy comparison, subtract that margin
and compute a paired one-sided sign-flip $p$ value for a mean below zero. Apply
Holm correction separately to the family of two force superiority tests and
the family of two energy noninferiority tests. Each adjusted $p$ value must be
below 0.05.
The strong absolute targets remain force RMSE at most $10^{-3}$ eV per angstrom
and energy error at most $10^{-4}$ eV per atom. Matching G0 is not a positive
compression result. Cross-checkpoint transfer, interpretability, and physical
simplicity are separate follow-up claims and cannot rescue a failed primary T8
outcome.

## Exactness and claim boundaries

- The symmetry of every projector remains exact.
- The $t=0$ tangent derivative remains exact for a fixed graph.
- The path identities remain exact for a fixed neighbor graph.
- Random force probes are used only for a spectral D1-EF estimator and require
  convergence tests. D2 obtains the force tangent from the exact scalar total
  derivative.
- Path quadrature is numerical and requires residual checks.
- Grassmann optimization does not certify a global optimum.
- The D1 matrices minimize a declared additive upper bound with explicit
  block-separation constants. They do not minimize the coherent D2 objective.
- D3 force optimization requires verified higher-order automatic
  differentiation through the complete injected graph.
- Neighbor-list changes at a cutoff require separate treatment.
- Discovery and evaluation geometries define an empirical measure.
- No result may be called classical ODT unless it reduces to the corresponding
  top-down environment and spectral truncation construction.
