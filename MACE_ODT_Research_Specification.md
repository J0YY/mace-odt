# Global Weight Decomposition of Pretrained MACE
## An ODT-grounded research specification for gauge-invariant, physically inspectable functional subspaces

**Document type:** Standalone research and implementation specification

**Date:** 21 September 2026

**Status:** Proposed project, with finite-dimensional mathematical checks; no new MACE experiments reported

**Primary sources:** Dooms et al., *Compositionality Unlocks Deep Interpretable Models*, arXiv:2504.02667v1; David Olloqui, *Reading the inside of a machine-learned interatomic potential*, supplied HTML manuscript
**Working name:** MACE-ODT. This is a project name, not a claim that the original ODT algorithm already supports every construction below.

---

## Navigation

The conceptual and source foundation is in Sections 1–6. The central construction and its proofs are in Sections 7–14. Full-model extensions are in Sections 15–16. Implementation and empirical evaluation are in Sections 17–23. Paper design, novelty, deliverables, and source locators are in Sections 24–27. The appendices document executed mathematical checks and the literal source-bound audit.

- [1. Scientific motivation and contribution boundary](#section-1)
- [2. Exact provenance: what comes from Dooms' χ-net paper](#section-2)
- [3. Exact provenance: how the HTML determines the project](#section-3)
- [4. Hypotheses, result tiers, and falsifiable claims](#section-4)
- [5. Objects, notation, and architectural scope](#section-5)
- [6. Coefficient lifts, physical functions, and identifiability](#section-6)
- [7. Input metrics and a precise definition of weight-only discovery](#section-7)
- [8. The two-sided bond decomposition: exact statement and limits](#section-8)
- [9. Primary algorithm: exact mixed-order decomposition of the first energy branch](#section-9)
- [10. Efficient contractions without dense global tensors](#section-10)
- [11. Quotienting architectural redundancy correctly](#section-11)
- [12. From canonical modes to invariant physical objects](#section-12)
- [13. The body-order bridge: exact formulas and limits of attribution](#section-13)
- [14. Error guarantees: what is bounded and what is not](#section-14)
- [15. Extending through both message-passing layers](#section-15)
- [16. Two distinct routes to the original nonlinear head](#section-16)
- [17. Implementation architecture and interfaces](#section-17)
- [18. Validation gates and test suite](#section-18)
- [19. Empirical program](#section-19)
- [20. Baselines and matched controls](#section-20)
- [21. Metrics and statistical reporting](#section-21)
- [22. Adversarial review and rejection criteria](#section-22)
- [23. Milestones, dependencies, and scope decisions](#section-23)
- [24. Paper design and figures](#section-24)
- [25. Novelty audit and relation to adjacent work](#section-25)
- [26. Deliverables and definition of completion](#section-26)
- [27. Source ledger](#section-27)
- [Appendix A. Executed mathematical checks](#appendix-a)
- [Appendix B. A concrete audit of the printed ODT truncation condition](#appendix-b)
- [Appendix C. Compact notation and claim checklist](#appendix-c)

---

## Executive overview

### The research question

Do pretrained MACE potentials contain substantially lower-dimensional **composed interaction structure** than is visible in individual weight matrices or in the radial functions entering a layer? Can that structure be extracted without fitting to molecular activations, represented independently of admissible hidden-coordinate changes, and connected to explicit physical interaction functions?

The intended result is not a new matrix factorization that supersedes singular value decomposition. It is a method for identifying, constructing, and efficiently decomposing the **right global operator**, together with evidence that its dominant subspaces preserve meaningful model behavior better than locally selected subspaces.

Dooms et al. supply the central computational principle: unfold a compositional polynomial network into a tensor network; orthogonalize the input-side subtrees; contract the output-side environment; diagonalize the resulting bond operators; and truncate the selected subspaces. The HTML supplies the MACE-specific scientific problem and much of the solution's infrastructure: explicit gauge freedoms, an exact coupling-path null space, radial images of channel covectors, finite body-order formulas for the first readout, and unusually strong controls against misleading interpretation. [D:S2–S4; D:AF–AG; H:gauge; H:nullspace; H:curves; H:expansion; H:synthesis]

The project has three distinct levels:

1. **Exact first-branch study.** Decompose the first-layer, linear-energy branch of a released MACE-OFF23 checkpoint. Treat its learned radial functions exactly as functions, up to controlled numerical integration; quotient the known path redundancy; and use equivariant, tied, mixed-order tensor projections. This is the primary tractable target.
2. **Full polynomial computation study.** Extend the analysis through both message-passing layers to polynomial observables that feed the original nonlinear head. This requires explicit treatment of shared occurrences, topology, and the chosen output metric.
3. **Full nonlinear-energy study.** Either keep the nonlinear head exact and bound the effect of changes to its preactivations, or introduce a separately certified polynomial shadow of the head. These are different methods and must be reported separately.

A successful first branch does not establish a decomposition of the complete potential. A small coefficient-spectrum effective dimension does not establish a correspondingly small fidelity-preserving rank. A chemically recognizable radial image does not, on its own, establish a chemical mechanism.

### The hoped-for finding

The strongest possible empirical finding would be that a **nontrivial, pre-product channel interface**, not merely a scalar output bottleneck, admits a small retained subspace that:

- is selected using the composed weights and declared structural measures, without molecular-activation fitting;
- survives admissible gauge transformations as the same functional subspace;
- preserves energy differences and forces at substantially lower rank than matched local-SVD alternatives;
- supports compact, reproducible descriptions in element-pair, radial, angular, and body-order coordinates;
- has intervention effects that are not explained by edit support, scale, an arbitrary dual, or a trivial linear-readout identity.

A hypothetical reduction from 96 channels to roughly 10–15 modes is a motivating example, **not a forecast, target to tune toward, or source result for MACE**. The rank must be measured separately for each interface, irrep block, central-element conditioning, norm, and fidelity criterion.

### What is already established versus proposed

| Category | Meaning in this specification |
|---|---|
| **[SOURCE-D]** | A statement made by the supplied Dooms paper, with a precise source locator. It is not automatically a theorem about MACE. |
| **[SOURCE-H]** | A statement or measurement reported in the supplied HTML. Its experiments have not been independently rerun here. |
| **[DERIVATION]** | A mathematical statement derived in this specification under explicit assumptions. It need not be new to the tensor-decomposition literature. |
| **[PROPOSAL]** | An algorithm, experimental protocol, or implementation design to build and evaluate. |
| **[HYPOTHESIS]** | An empirical outcome whose failure is allowed and informative. |
| **[OPEN]** | A necessary extension or proof obligation that is not resolved by the sources or the present derivation. |
| **[EXTERNAL]** | Supplementary primary-source material consulted to verify an architectural or mathematical point. |

Source identifiers such as `[D:AG]` and `[H:curves]` are resolved in the source ledger at the end. The ledger uses sections, equation numbers, figure numbers, HTML anchor IDs, and original source-line ranges rather than chat-dependent references.

---

<a id="section-1"></a>

## 1. Scientific motivation and contribution boundary

### 1.1 Why local weight SVD is insufficient

Consider a true linear separator with upstream map

$$
U:\mathcal A\to H
$$

and downstream map

$$
V:H\to\mathcal B.
$$

The composed operator is $T=VU$. A direction with large upstream gain can be nearly annihilated downstream; a direction with tiny upstream gain can be amplified downstream. For example,

$$
U=\operatorname{diag}(100,0.01),\qquad
V=\operatorname{diag}(0.0001,100),
$$

so

$$
VU=\operatorname{diag}(0.01,1).
$$

Keeping the leading direction of $U$ incurs a composed Frobenius error of 1. Keeping the leading direction of $VU$ incurs an error of 0.01. This is a synthetic illustration, not a MACE result. [DERIVATION; mathematical check `local_vs_composed_svd`]

The analogue in a tensor program is not usually the product of two ordinary layer matrices. It is a matricization of the full coefficient tensor across a specified partition of its external legs, represented implicitly by a tensor-network separator.

### 1.2 What ODT adds to SVD

ODT does not increase SVD's expressive power. It makes a family of otherwise enormous **global tensor matricizations** accessible through small bond-space calculations. It exploits the tree structure, isometric cancellations, and tied symmetric cores of a χ-net. [D:S3; D:AF–AG]

The distinction is therefore:

- local SVD: decompose one parameter tensor;
- radial-curve SVD: decompose the map from channel coefficients to local radial functions;
- global cut decomposition: decompose the function tensor across an identified separator;
- full physical-behavior approximation: preserve the original model on molecular configurations and their coordinate derivatives.

These are related but different optimization problems. The project is valuable only to the extent that the global construction improves useful outcomes beyond its own built-in coefficient-norm objective.

### 1.3 Why the HTML is central rather than ancillary

The HTML is not simply a source of pictures for labeling ODT modes. It constrains the project's objects, algebra, intervention definitions, controls, and success criteria.

Its central methodological demand is that an interpretation survive the relevant coordinate freedoms and controls that remove the proposed explanatory content. Its radial-curve results show that shared geometric representation and shared benchmark effect can diverge. Its removal-rule experiments show that a direction without a specified dual is not a complete intervention. Its body-order analysis shows both an exact tractable branch and a nonlinear branch with strong cancellations. [H:problem; H:synthesis; H:comparing; H:deflation; H:calibrations; H:appG-conformers]

Consequently, the project should preserve the HTML's distinctions, not replace its negative controls with a claim that eigenvectors are inherently interpretable.

### 1.4 Contributions that could be defensible

A complete paper may make some or all of the following contributions, depending on results:

**Construction.** A checkpoint-derived, equivariant tensor representation of selected MACE computations, with explicit separation of exact algebraic identities and numerical approximations.

**Decomposition.** An implementation of ODT-style global environments for valid separators, and a tied mixed-order projection procedure for the first-layer polynomial branch.

**Identifiability.** Gauge-covariant encoders, decoders, and projectors, with invariant functional images and clear limits on the gauge family covered.

**Scientific finding.** Evidence for, or against, useful compositional low rank beyond known architecture-forced redundancy.

**Interpretation.** A controlled connection between extracted subspaces, physical interaction kernels, and model-specific chemical effects.

**Engineering.** A measured reduction in inference cost, storage, or memory, if a compiled compressed evaluator actually provides it.

The two-sided cut identity, ordinary eigendecomposition, whitening, partial traces, and general HSVD principles are not themselves new contributions. A practical specialization and its empirical scientific findings may be.

---

<a id="section-2"></a>

## 2. Exact provenance: what comes from Dooms' χ-net paper

### 2.1 Source identity and scope

The source is the ten-page arXiv v1 paper by Thomas Dooms, Ward Gauderis, Geraint A. Wiggins, and José Oramas, dated 3 April 2025. The supplied `chi-nets.txt` is a text rendering of this paper. The PDF's diagrams and equations were also inspected because the text extraction loses tensor wires and superscripts. [D]

The paper introduces an architecture and an algorithm together. It does not provide an off-the-shelf theorem or implementation for arbitrary MACE graphs, arbitrary neural nonlinearities, or arbitrary hidden-space Gram matrices.

### 2.2 Lineage table

| Item inherited from Dooms | Exact location | What the project inherits | What it does not inherit automatically |
|---|---|---|---|
| Nonlinear cloning and unfolding | §2, p.2, Eq.(1), Fig.1 | Repeated use of the same vector can be represented by independent tensor slots evaluated on repeated inputs. | That a folded nonlinear graph is a linear map on its original input space. |
| χ-net architecture | §2, p.2 | Linear embedding, repeated bilinear cores, linear unembedding. | A claim that MACE has precisely this topology. |
| Factorized bilinear cores | §2, Eq.(2) | Bilinear maps can have efficient factored implementations. | Closure of the original factorization class under arbitrary canonicalization and truncation. |
| Constant-coordinate lifting | §2, p.2 | Appending a constant permits lower-degree terms within a homogeneous lift. | A unique coefficient norm independent of the constant-coordinate convention. |
| Local input-slot symmetrization | Start of p.3 | A bilinear core can be symmetrized because both slots receive the same vector. | Full permutation symmetry of all leaves of a deep unfolded tree. |
| Bottom-up orthogonalization | §3; Appendix G, Algorithm 1; Fig.10 | RQ each core and absorb the non-isometric factor in its consumers. | The same recursion for arbitrary residual, graph, or multi-consumer structures. |
| Absorption of $R_i\otimes R_i$ | Algorithm 1, line 7 | A factor must be applied to each cloned input slot. | A reason to explicitly allocate a Kronecker-power array. |
| Global environment diagonalization | §3; Algorithm 2; Fig.11 | Contract the rest of the doubled network and diagonalize a small bond operator. | A raw-coordinate Gram that is invariant under general invertible gauges. |
| Projector insertion | §3, Eq.(3); Algorithm 3; Figs.13–14 | Use retained eigenvectors to reduce bond dimensions in adjacent cores. | Identical native MACE parameter counts or faster execution. |
| Tree-specific efficiency | Appendix F, p.7 | Exploit symmetry and repeated isometric subtrees. | An $O(Lh^4)$ bound for general MACE programs. |
| Tensor-norm compression discussion | §3, p.3 | Analyze coefficient-tensor error and distinguish it from task loss. | A direct energy, force, or molecular-dynamics guarantee. |
| SVHN empirical results | §4; Appendix E; Tables 2–3; Figs.2,7–9 | A motivating example where composed structure is much more compact than local spectra suggest. | A numerical prediction for MACE. |
| Weight interpretation | §4, p.4; Figs.3–4; Appendices C–D | Inspect both basis functions and their interactions. | That every retained mode is an independent additive or human-semantic mechanism. |

### 2.3 The χ-net in explicit notation

Let $\bar x=(1,x)$ when a constant coordinate is used. Write

$$
h_1=e\bar x,
\qquad
h_{i+1}=F_i[h_i\otimes h_i],
\qquad
y=uh_{L+1}.
$$

A core has entries

$$
(F_i)_{oab},\qquad
F_i\in\mathbb R^{h_{i+1}\times h_i\times h_i}.
$$

The factored bilinear parameterization in Eq.(2) corresponds, before symmetrization, to

$$
(F_i)_{oab}=(A_i)_{oa}(B_i)_{ob}.
$$

The core can be replaced by

$$
(F_i)_{oab}\leftarrow
\frac12\big[(F_i)_{oab}+(F_i)_{oba}\big]
$$

without changing its repeated-input evaluation. [D:S2]

Unfolding yields a tensor $\Theta$ such that

$$
y=\Theta[\bar x^{\otimes 2^L}].
$$

The multilinearity is in the **separate unfolded slots**. On the original repeated input $x$, the model is polynomial, not multilinear in general.

### 2.4 Orthogonalization, including matrix orientation

Reshape each current core as an output-by-input matrix and factor

$$
F_i^{\mathrm{mat}}=C_iQ_i,
\qquad Q_iQ_i^\top=I
$$

on its supported output space. Here $C_i$ is the non-isometric RQ factor; a new letter avoids confusing it with a downstream environment.

Replacing $F_i$ by $Q_i$ requires replacing its parent by

$$
F_{i+1}\leftarrow F_{i+1}(C_i\otimes C_i).
$$

At the top, absorb $C_L$ into $u$. This is the algebra of Algorithm 1. QR of the transpose is an equivalent implementation of the factorization, provided orientations and rectangular supports are handled correctly. [D:AG, Algorithm 1]

### 2.5 The top-down environment contains a partial trace

Initialize

$$
G_{L+1}=u_\perp^\top u_\perp.
$$

For an isometric binary core $Q_i$, the environment on one child slot is

$$
(G_i)_{aa'}
=
\sum_{o,o',b}
(Q_i)_{oab}
(G_{i+1})_{oo'}
(Q_i)_{o'a'b}.
\tag{D1}
$$

The sibling index $b$ is contracted. Equivalently,

$$
G_i=\operatorname{Tr}_{\mathrm{sibling}}
\left(Q_i^\top G_{i+1}Q_i\right).
$$

Without the sibling trace, ordinary matrix multiplication produces an operator on $H_i\otimes H_i$, not on $H_i$. Algorithm 2's compact diagrammatic notation must not be translated into an incorrectly shaped matrix product. Symmetry makes the two child environments equivalent in the χ-net setting. [D:AG, Algorithm 2 and Fig.11]

### 2.6 Diagonalization and truncation

Write

$$
G_i=Z_i\Lambda_iZ_i^\top,
\qquad \lambda_{i,1}\ge\lambda_{i,2}\ge\cdots\ge0.
$$

For a retained rank $r_i$, use $Z_{i,r_i}$, whose columns are orthonormal, and insert

$$
P_i=Z_{i,r_i}Z_{i,r_i}^\top.
$$

The parent core receives $Z_{i,r_i}$ on each child input, while the lower core receives $Z_{i,r_i}^\top$ on its output. In a folded implementation this means the same reduced coordinates must be used consistently wherever the logical hidden state occurs. [D:S3; D:AG, Algorithm 3]

### 2.7 What the reported numbers mean

Dooms' Table 3 reports the following **effective $L_2$ bond dimensions**:

| Source label | Bond 0 | Bond 1 | Bond 2 | Bond 3 |
|---|---:|---:|---:|---:|
| SVD | 83.4 | 165.9 | 100.5 | 229.2 |
| ODT | 2.1 | 3.3 | 13.3 | 5.7 |

These are source-reported effective dimensions, not the retained ranks at which test accuracy is preserved. Figure 2 separately reports that approximately 70% of model dimensions can be removed without an accuracy decrease. The specification does not identify the Table 3 metric with participation ratio or stable rank unless the exact source implementation confirms that definition. [D:AE; D:Table3; D:Fig2]

The study uses a three-layer SVHN model with a 256-dimensional embedding and ten outputs. Table 2 gives 85.4% for the three-layer χ-net and 87.3% for its ReLU baseline. The abstract's broad accuracy language should not substitute for those measurements. [D:AA–AB]

Appendix E also reports that excluding the constant direction raises the effective dimension to approximately 14 across layers. The small numbers therefore need constant-coordinate and output-rank controls. [D:AE]

### 2.8 Mathematical audit of the source's compression notation

The paper diagonalizes a Gram matrix but describes its diagonal entries as singular values. In conventional matrix notation, if

$$
G=MM^\top,
$$

then its eigenvalues are $\lambda_j=\sigma_j(M)^2$. The squared residual for the rank-$r$ SVD of $M$ is

$$
\sum_{j>r}\sigma_j^2=\sum_{j>r}\lambda_j,
$$

not $\sum_{j>r}\lambda_j^2$.

The displayed §3 sufficient condition is written using differences of squared Frobenius norms of Gram matrices. Read literally, it controls a fourth-power singular-value tail. This specification does **not** adopt that condition as an executable certificate. It uses independently derived discarded-eigenvalue sums and explicit tree-occurrence counting. A generic counterexample and an explicit one-layer χ-net test of the literal displayed condition are included in the mathematical checks and Appendix B. These identify a notation/proof obligation; it is not a claim that every empirical ODT result is invalid. [D:S3, p.3; DERIVATION]

The paper also states that truncation retains isometric properties. Cropping the input legs of a row-isometric core generally does not preserve all its row norms. Re-orthogonalize a truncated network before using identities that require isometric subtrees. Preservation of the represented function before truncation, and correctness of the inserted projectors, are separate issues from post-truncation canonical form. [D:S3; DERIVATION]

---

<a id="section-3"></a>

## 3. Exact provenance: how the HTML determines the project

### 3.1 Architecture and the two readouts

The HTML studies the released MACE-OFF23 small, medium, and large family. It reports a small model with 694,320 parameters, 96 channels, two message-passing layers, and a 4.5 Å cutoff. Its key energy identity is

$$
E_i=E_0(z_i)+s\left[w^\top h_i^{(1)}+g(h_i^{(2)})\right]+b.
\tag{H1}
$$

The first readout is linear; the final readout is a one-hidden-layer MLP. Appendix A describes sixteen hidden units in that head. These are source descriptions to verify against the exact loaded checkpoint, not universal facts about every current MACE variant. [H:model, Eq.(1); H:appA, Eq.(15)]

**Project use:** Every experiment must record which term in (H1) is targeted. The original atomic references, scale, and offsets are preserved exactly. A decomposition of $w^\top h^{(1)}$ cannot be described as a decomposition of $g(h^{(2)})$.

### 3.2 The exact gauge family

At a particular layer-1 linear interface, the source identifies

$$
A\mapsto MA,
\qquad
B_z\mapsto B_zM^{-1},
\tag{H2}
$$

so that $B_zA$ is unchanged. The source distinguishes this interface from later channel spaces where a channel-diagonal convolution restricts architecture-preserving transformations. [H:gauge, Eq.(2); H:appB-coverage]

**Project use:** Build an interface registry rather than a single generic “layer-1 channel” label. Every mode must identify its exact producer, consumers, irrep block, conditioning, admissible transformations, and intervention location.

A raw angular-block Gram is not generally invariant under an arbitrary $GL(c)$ transform. Invariance must be demonstrated for the specific composite object and its metrics. Broad language about a “Gram or spectrum” in the HTML does not license an ordinary raw-channel spectrum without this check. [DERIVATION based on H2]

### 3.3 The exact contraction null space

The source defines

$$
\mathcal T:W\mapsto
\operatorname{Sym}\left(\sum_\eta W_\eta\mathcal C_\eta\right).
\tag{H3}
$$

For its scalar correlation-three contraction, 23 stored paths map to an eight-dimensional image, leaving a 15-dimensional kernel. Appendix C reports analogous counts of $51\to12$ for vector output and $65\to14$ for rank-two output. [H:nullspace, Eq.(3); H:appC-derivation]

**Project use:** Quotient these coefficient redundancies before attributing any remaining compactness to training. Treat this as **path-space reduction**, not evidence that fifteen channel directions must disappear from a channel-bond spectrum.

The HTML reports that reducing stored coefficients and the accompanying coupling tensors preserves energies and forces to numerical precision and changes actual inference cost. Merely zeroing a stored coefficient is not the same operation. [H:appC-conversion]

### 3.4 The radial-curve image

The source writes the first interaction block as

$$
X_i^{(\ell)}=
\sum_{j\in\mathcal N(i)}
F_{z_iz_j}^{(\ell)}(r_{ij})
Y_\ell(\widehat r_{ij})^\top.
\tag{H4}
$$

A channel **covector** $d$ has radial image

$$
f_d(z,z',r)=d^\top F_{zz'}^{(\ell)}(r).
\tag{H5}
$$

When $F\mapsto MF$ and $d\mapsto M^{-\top}d$, the image is unchanged. The source obtains the curves from two-atom evaluations and reports a block-reconstruction residual of approximately $2.1\times10^{-15}$. [H:curves, Eqs.(7)–(8)]

**Project use:** This is simultaneously an upstream function-space construction, a way to define deterministic structural inner products, a gauge-invariance test, and a semantic representation of **encoder** modes. It is not automatically a map from every downstream eigenvector or SAE decoder vector to a physical function.

### 3.5 The local-mode failure that motivates the global comparison

The source's radial-curve SVD is dominated by a few modes. It reports a leading singular value 78 times the 45th. Projecting the selected directions into the first five curve modes produces a pairwise cosine of approximately 0.92, but several corresponding deletions worsen the benchmark. At the sampled 45-mode rung, their effects have largely returned and their pairwise cosine is approximately 0.39. [H:deflation, Fig.10, Table3]

**Project use:** Compare local geometric amplitude with composed coefficient importance and with measured intervention effect. The observation motivates the comparison but does not show that ODT will recover the benchmark-improving directions.

The ladder jumps from 20 to 45. It does not establish an exact transition at 45, and 45 is not the known minimal rank of the MACE branch. [H:limitations]

### 3.6 Direction plus dual defines an intervention

The source contrasts

$$
B_z\leftarrow(I-dd^\top)B_z
$$

with

$$
B_z\leftarrow
\left(I-\frac{dw^\top}{w^\top d}\right)B_z.
\tag{H6}
$$

It reports substantial changes in apparent benchmark gains when the removal rule changes. The four SAE directions have approximately 12.4–16.3% gains under orthogonal removal, but at most 0.88% under the reported covariance-dual removal. The contrast direction has 3.89% versus 15.15% under the two rules. [H:comparing, Eq.(9); H:calibrations]

**Project use:** Export encoder/decoder pairs and the resulting oblique native projector. Never report a deletion direction alone. A comparison with the HTML's interventions must match the dual or explicitly study it as a separate factor.

### 3.7 The three negative controls

The HTML's detection, ablation, and steering tests are non-diagnostic **under the controls studied there**. The source explicitly does not infer that SAE latents are non-causal. [H:synthesis]

**Detection control:** A random embedding of the label table reproduces the kind of label wins otherwise attributed to sparse features. **Project use:** Chemical-label separability is a secondary diagnostic, not primary mechanism evidence.

**Ablation control:** For a binary reach indicator,

$$
\mathrm{AUC}_R=
\frac12+\frac12\left[
\Pr(R\mid\mathrm{match})-
\Pr(R\mid\mathrm{other})\right].
\tag{H7}
$$

This is the reach-only statistic, not a universal formula for the full magnitude-sensitive AUC. **Project use:** Match edit support and amplitude, and separately report reach and conditional magnitude discrimination. [H:selectivity; H:appF]

**Steering control:** The linear-readout response is

$$
\Delta E_i=s\,w^\top u\,\delta.
\tag{H8}
$$

**Project use:** Subtract or report this known contribution before attributing nonlinear chemical specificity. [H:steering]

### 3.8 Reconstruction and force fidelity

The source reports 267 meV/site of energy-readout reconstruction error for its chosen SAE protocol, compared with conformational differences of 25–48 meV, and large changes to forces for some benchmark-improving edits. [H:appE-resolution; H:calibrations]

**Project use:** Feature reconstruction alone is inadequate. Evaluate relative conformer energies, per-coordinate force errors, tail behavior, and the perturbation relative to the unmodified model's own error. Improving one path benchmark while damaging equilibrium forces is not unqualified success.

### 3.9 Body-order expansion and its limits

For a central site and a neighbor subset $S$, the source uses

$$
V_i(S)=\sum_{T\subseteq S}
(-1)^{|S|-|T|}e_i(T),
\qquad
e_i(\mathcal N_i)=\sum_{S\subseteq\mathcal N_i}V_i(S).
\tag{H9}
$$

For the first, linear-energy branch, terms with four or more neighbors vanish because each polynomial monomial involves at most three neighbor selections. The physical body order includes the central atom, so the branch is at most four-body. [H:expansion; H:appG-expansion, Eqs.(25)–(26)]

**Project use:** This provides an exact decoder from reduced polynomial kernels to pair, triplet, and quadruplet contributions, as well as a stringent vanishing test. Correlation order and distinct-atom body order must not be conflated.

The source warns that isolated fragments can be extrapolative. Its nonlinear-head terms can be much larger than their summed contribution, with strong cancellation. [H:appG-conformers, Eq.(27)]

**Project use:** Exact decomposability does not make each term a stable physical attribution. Report cancellation and geometry domain.

### 3.10 Concrete chemistry and where it is located

For 1,2-difluoroethane, the source reports the following gauche-minus-anti contributions:

| Contribution | Small | Medium | Large |
|---|---:|---:|---:|
| Linear readout | +35.1 meV | +22.9 meV | +20.1 meV |
| MLP readout | −59.9 meV | −62.8 meV | −62.5 meV |

The linear branch has the opposite sign from the total preference. Thus a first-branch decomposition cannot by itself explain the full preference. The source places the ethane barrier largely in the small model's linear branch, making that a more suitable first-branch behavioral case study, subject to independent reproduction. [H:appG-conformers]

The HTML also reports that learned pair and angular terms are not uniformly aligned with textbook bond references, and that element identity interacts across weight blocks rather than separating additively. **Project use:** Do not assume that each important mode corresponds to one element pair or that editing one element-indexed block isolates that element's contribution. [H:terms; H:appG-elements]

### 3.11 Source-quality and availability requirements

The HTML is a supplied research manuscript, not an independently reproduced experimental record in this project. Its text contains unresolved details that must be checked before an exact replication claim:

- The exact checkpoint, package revision, hook locations, SAE artifacts, and selected covectors/decoders are not fully specified by the prose alone.
- Broad references to “layer 1” cover more than one interface with different gauge restrictions.
- Its image formula uses covector transport, while its removal formula uses a perturbation direction and dual; those roles require explicit implementation-level reconciliation.
- The paragraph describing all five selected directions' gains is broader than Table 3, whose contrast-direction gain is 3.89%. Use the table's individual entries rather than assigning 12.4–16.3% to all five.
- Its broad claims about Gram/spectral invariance must be read relative to the specific constructed invariant, not arbitrary raw-coordinate Grams.
- Reported optimizer behavior is relevant to optional fine-tuning, but an adaptive optimizer is not invariant to arbitrary coordinate changes. No training-equivalence theorem is imported from a conversion identity.

These requirements preserve the distinction between source reports, verified identities, and new experiments.

---

<a id="section-4"></a>

## 4. Hypotheses, result tiers, and falsifiable claims

### 4.1 Main hypotheses

**H1 — Compositional compactness.** After exact path-nullspace reduction, some nontrivial MACE interfaces have faster global coefficient-spectrum decay than matched local decompositions.

**H2 — Fidelity advantage.** At matched retained multiplicity, the global subspaces preserve the targeted model computation better than local radial-SVD, local core-SVD, and appropriate random subspaces on unseen configurations.

**H3 — Identifiability.** The retained functional subspaces, images, and consistently transported interventions are invariant under the explicitly tested architecture-preserving gauge family, up to degeneracy and numerical conditioning.

**H4 — Physical organization.** Some dominant subspaces admit simpler or more stable radial, species, angular, or body-order descriptions than matched controls.

**H5 — Mechanistic specificity.** At least one such description predicts a nontrivial behavioral intervention effect beyond support, scale, and readout-only explanations.

**H6 — Extension.** A tractable subset of the full two-layer computation retains these properties when the nonlinear head's dependencies are included.

H1 does not imply H2; H2 does not imply H4; H4 does not imply H5; success on the first energy branch does not imply H6.

### 4.2 Result tiers

| Tier | Evidence required | Appropriate description |
|---|---|---|
| 0 | Exact quotient and forward equivalence | Architectural redundancy removal; mostly a prerequisite. |
| 1 | Correct global spectra on nontrivial interfaces | Structural decomposition under a declared norm. |
| 2 | Rank–fidelity advantage on held-out molecules | Useful functional compression of the stated target. |
| 3 | Gauge-stable, readable kernels with controlled interventions | Evidence for model-specific interpretable interaction subspaces. |
| 4 | Full-model target plus derivative fidelity and practical compilation | An extension of ODT-style analysis to a pretrained nonlinear potential. |

Tier 1 alone may be mathematically correct yet scientifically incremental. A strong interpretability paper should aim for Tier 3, with Tier 4 a valuable extension rather than an assumed outcome.

### 4.3 Interpretation of possible outcomes

A sharp spectrum with poor force fidelity means the chosen structural norm is not a good proxy for the intended physical behavior. A flat spectrum after quotienting does not prove that no interpretable mechanisms exist; it rejects a particular low-rank hypothesis for the tested interfaces and measures. A compact but chemically mixed subspace supports compression, not automatically mechanistic interpretation. Good performance only under a task-tuned measure supports a task-conditioned method, not a strictly task-agnostic discovery claim.

Comparisons with random initialization are necessary before attributing compactness specifically to learning. Constant offsets, scalar terminal ranks, known nullspaces, factorization constraints, and the chosen basis can all create compactness without a learned mechanism.

---

<a id="section-5"></a>

## 5. Objects, notation, and architectural scope

### 5.1 Three parameter and function spaces that must stay separate

There are at least three relevant spaces:

$$
\mathcal P=\text{stored parameter space},
\qquad
H_b=\text{a hidden channel/multiplicity space},
\qquad
\mathscr F=\text{a space of functions of geometry}.
$$

The path-nullspace map acts on a subspace of $\mathcal P$. A bond projector acts on $H_b$. The radial image maps an element of $H_b^*$ to $\mathscr F$. Their dimensions and null spaces are not interchangeable.

A fourth object, the coefficient tensor $\Theta$, belongs to a tensor product of **external feature-function spaces**, not automatically to a product containing an arbitrary internal bond as a free index.

### 5.2 Required interface registry

For every candidate interface, store:

```text
interface_id
checkpoint_hash
producer_module_path
consumer_module_paths
position_relative_to_element_map_and_product_basis
layer_index
irrep_label = (ell, parity)
multiplicity
central_species_conditioning
logical_state_id
unfolded_occurrence_ids
native_architecture_gauge_family
analysis_tensor_gauge_family
target_observables
intervention_scope = branch_only | all_consumers | occurrence_only
```

A full $GL(c)$ change on an expanded tensor-network bond can be a legal analysis reparameterization even when it is not realizable as a checkpoint edit within the original channel-diagonal MACE architecture. Those are different gauge families and must have different tests.

### 5.3 MACE operations in the primary scope

For the specific OFF23 family described in the HTML, the first-branch compiler must account for:

1. element embeddings and every relevant element-conditioned map;
2. learned radial functions and cutoff factors;
3. spherical harmonics and their exact normalization convention;
4. neighbor aggregation and fixed normalization constants;
5. channel maps feeding the product basis;
6. symmetric products of correlation orders one, two, and three;
7. fixed CG tensors and learned path coefficients;
8. post-product linear maps and any residual term;
9. the first-layer scalar readout and the checkpoint scale.

The radial functions can themselves contain nonlinear MLPs. Calling the subsequent computation polynomial means **polynomial in these primitive functions or their aggregate features**, not polynomial in interatomic distance or Cartesian coordinates. [H:appA; EXTERNAL M:code]

Current MACE source contains additional interaction variants, including density normalization and feature-dependent nonlinear operations. The compiler must inspect and classify the loaded model, rather than accept every module named MACE. An unsupported node is a hard diagnostic, not an invitation to silently replace it with a linear approximation. [M:code]

### 5.4 Polynomial branch representation

Fix a central species $z$, and let a neighbor primitive be

$$
t=(z',r,\Omega),\qquad \Omega\in S^2.
$$

Let $\psi_z(t)$ collect the relevant channel–angular features after the precisely registered linear interface. The aggregate is

$$
a_z(X)=\sum_{t_j\in X}\psi_z(t_j).
$$

After composing the first branch's learned maps, CG contractions, and scalar readout, write

$$
E_{L,z}(X)
=c_z+\sum_{\nu=1}^{3}
T_{z,\nu}\left[a_z(X)^{\otimes\nu}\right].
\tag{1}
$$

Here each $T_{z,\nu}$ is symmetric in its repeated input slots. The representation is exact for the scoped branch when all its operations have been compiled correctly. If a registered intermediate basis needs multiple radial families, use a direct sum of those families rather than dropping them.

The constant $c_z$ includes only the branch's geometry-independent contribution. Keep the separate atomic reference energies and other fixed offsets from (H1) in their own exact path.

### 5.5 Conditional versus shared spaces

The primary implementation analyzes each central species separately. This keeps the input slots conditionally independent under a declared product measure and permits a non-ambiguous whitening.

A collection of per-species rank-$r$ projectors is not the same as one shared native rank-$r$ projector. Report:

$$
r_{z,\ell},\quad
r^{\max}_\ell=\max_z r_{z,\ell},\quad
\text{total stored coefficients},\quad
\text{actual evaluator cost}.
$$

Do not multiply independently sampled central-species labels across cloned slots. All neighbors of one site have the same central species. A joint construction across central species must retain that constraint explicitly.

---

<a id="section-6"></a>

## 6. Coefficient lifts, physical functions, and identifiability

### 6.1 Three notions of “the same model”

A decomposition can be invariant under progressively stronger equivalences:

**Tensor-network gauge equivalence:** Different factorizations of the same fixed external coefficient tensor.

**Scoped architectural equivalence:** Checkpoints related by the explicitly derived MACE transformations and nullspace changes.

**Full physical-function equivalence:** Any checkpoints producing the same physical energy and forces on every admissible atomic configuration.

The first does not imply the third. A formal coefficient lift can contain directions that disappear when its separate slots are evaluated on repeated or geometrically constrained inputs.

### 6.2 A concrete deep-lift ambiguity

Consider the polynomial identity

$$
x_1^2x_2^2-(x_1x_2)^2=0.
$$

It has a two-level compositional realization: first compute $x_1^2$, $x_1x_2$, and $x_2^2$, then combine them with a symmetric quadratic output.

Its unfolded fourth-order tensor can be nonzero even though its diagonal evaluation is identically zero. One example has

$$
\Delta_{1122}=\Delta_{2211}=\frac12,
$$

and

$$
\Delta_{1212}=\Delta_{1221}=\Delta_{2112}=\Delta_{2121}=-\frac14,
$$

with all other entries zero. It is symmetric within the two child pairs and under exchanging those pairs, but it is not fully symmetric over all four leaves. Its norm is $\sqrt{3/4}$, while

$$
\Delta[x^{\otimes4}]=0
$$

for every $x$. Full symmetrization makes it zero. [DERIVATION; check `tree_lift_null_polynomial`]

Thus local core symmetrization does not make a deep tree's coefficient tensor a unique representation of the folded polynomial. ODT spectra of a chosen deep lift are not automatically invariants of the folded physical function under every algebraic rewrite.

### 6.3 What is fixed in the first-branch construction

For the primary correlation-three branch, explicitly symmetrize the complete $T_{z,\nu}$ over all $\nu\le3$ slots, after composing the relevant maps. At this order the symmetrization is manageable and aligns with the HTML's path-nullspace argument.

Fix:

- the primitive-function family;
- the central-species conditioning;
- the order convention;
- any removal of exact linear dependencies;
- the inner product;
- the definition of density-polynomial coefficients.

This removes a significant class of avoidable representation artifacts. It does not prove that the representation is unique modulo every identity satisfied by real atomic configurations or nonlinear radial functions.

### 6.4 Independent copies are not the physical repeated-input distribution

For a scalar $x$ uniformly distributed on $[-1,1]$,

$$
\mathbb E[x^4]=\frac15,
\qquad
\mathbb E[x^2]^2=\frac19.
$$

Contracting two separate quadratic input slots using two second moments gives the second number. Evaluating both slots at the same physical input and then taking the squared norm gives the first.

More generally, a physical polynomial norm needs higher-order moments such as

$$
\mathbb E[\phi(x)^{\otimes 2\nu}],
$$

not just a tensor power of $\mathbb E[\phi(x)\phi(x)^\top]$. [DERIVATION; check `physical_vs_independent_copy_metric`]

The primary coefficient norm is therefore a **structural product-space norm**. It is not quietly interpreted as the model's mean-squared energy error under an unspecified molecular distribution.

### 6.5 Off-manifold relationships and graph constraints

Atomic geometries couple distances and angles; neighborhood topology depends on cutoff membership; repeated neighbors contribute diagonal terms; and paths through a graph can share atoms and edges. An unfolded formal product space may include combinations that no physical configuration realizes.

There are two valid research choices:

- use the formal product-space norm, state its scope, and validate on real configurations;
- introduce a physically constrained metric, specify its measure, and accept that contractions may require high-order correlated objects.

The second is not automatically achievable with small independent upstream and downstream matrices.

---

<a id="section-7"></a>

## 7. Input metrics and a precise definition of weight-only discovery

### 7.1 Data-independence policy

The primary discovery pipeline may use:

- checkpoint parameters and stored architecture constants;
- exact algebraic CG identities;
- predetermined element and geometry domains;
- deterministic quadrature of checkpoint-defined radial functions;
- exact coefficient contractions and linear algebra;
- predetermined numerical tolerances.

It may not use molecular activation samples, task labels, quantum-reference residuals, or a benchmark-selected rank to choose modes in the primary task-agnostic arm.

Evaluating a learned radial network on a declared quadrature grid is a function evaluation derived from the checkpoint. It is not a training-distribution activation covariance. Nevertheless, describe the method as **checkpoint-derived with a declared structural measure**, not as literally requiring only a list of weights and no additional choices.

Molecular configurations are allowed for validation, held-out testing, and explicitly labeled empirical baselines. Data-assisted discovery is an optional separate arm.

### 7.2 The structural measure

For a fixed central species $z$, declare

$$
d\mu_z(t)
=\pi(z'\mid z)\,w_z(r)\,dr\,\frac{d\Omega}{4\pi}.
\tag{2}
$$

Use spherical harmonics normalized to be orthonormal under the chosen angular measure; convert library normalization factors explicitly.

The measure requires decisions about species weights, radial range, radial units, and radial weight. No source establishes one uniquely correct radial measure. Candidate preregistered measures can include uniform radial distance and volume weighting $r^2dr$, normalized over the same interval.

A suggested engineering default is equal neighbor-species weights and a dimensionless radial coordinate $r/r_c$, with the lower radius fixed before inspecting spectra. The physical interval must be stated explicitly. Alternative intervals are sensitivity analyses, not opportunities to select the best-looking spectrum.

### 7.3 Orthonormal radial functions from the HTML curves

At each $(z,\ell,p)$, form

$$
L_{z,\ell}
=
\sum_{z'}\pi(z'\mid z)
\int F_{zz'}^{(\ell)}(r)
F_{zz'}^{(\ell)}(r)^\top
w_z(r)\,dr.
\tag{3}
$$

Factor $L_{z,\ell}=C_{z,\ell}C_{z,\ell}^\top$ on its supported space and write

$$
F_{zz'}^{(\ell)}(r)
=C_{z,\ell}\,q_{zz'}^{(\ell)}(r),
\qquad
\langle q_a,q_b\rangle_{\mu_z}=\delta_{ab}.
\tag{4}
$$

This is function-space orthogonalization of the upstream radial map. It does not select the important modes; the downstream-composed kernel determines those later.

When $L$ is full rank, $q=C^{-1}F$. With exact rank deficiency, restrict to the functional support and distinguish an exact quotient from an approximate truncation of small singular values.

An exact zero eigenvalue of the ideal integral establishes a null function almost everywhere under the measure. Extending that statement to every point in the claimed domain requires suitable continuity and measure-support assumptions. A zero eigenvalue of a finite quadrature matrix is not, by itself, proof of an exact functional dependency. Distinguish algebraically verified null functions from numerically small directions and include the latter in the approximation budget.

### 7.4 Numerical stability

Avoid squaring condition numbers unnecessarily. A weighted quadrature feature matrix can be factorized directly using QR/SVD rather than first forming $FF^\top$. Use double precision in the reference path.

Record the quadrature order, convergence of every relevant overlap, supported rank, smallest retained metric eigenvalue, and condition number. Raw metric eigenvalue thresholding is not invariant to arbitrary ill-conditioned gauges in finite precision. Use support checks and comparisons in the common functional space.

Adding $\epsilon I$ to a raw-coordinate metric is not generally $GL(c)$-covariant. A regularizer must either be declared as a coordinate convention, be constructed from an explicitly transported metric, or be applied after canonicalization.

### 7.5 Order weights and output units

For coefficient tensors of different correlation orders, define

$$
\|\mathcal K\|_{\beta}^2
=\sum_{\nu=1}^{3}\beta_\nu\|K_{z,\nu}\|_F^2,
\qquad \beta_\nu>0.
\tag{5}
$$

The $\beta_\nu$ are part of the method. They affect which combinations of orders dominate. Primary experiments should report order-resolved spectra as well as an aggregate, and repeat a small preregistered set of order-weight conventions.

Do not silently use a direct-sum coefficient norm as though it included physical cancellation between different orders. Likewise, a vector-output metric requires an explicit relative scaling of its observables.

---

<a id="section-8"></a>

## 8. The two-sided bond decomposition: exact statement and limits

### 8.1 Assumptions

This section concerns a **genuine linear separator** of a fixed external coefficient tensor. After grouping external legs, suppose

$$
T=VU,
\qquad U\in\mathbb R^{c\times n_A},
\qquad V\in\mathbb R^{n_B\times c}.
\tag{6}
$$

The external coordinates are orthonormal under declared metrics. Define

$$
L=UU^\top,
\qquad R=V^\top V.
\tag{7}
$$

This formulation is standard finite-dimensional linear algebra underlying canonical tensor-network cuts. It is not a new general solution for nonlinear graphs.

### 8.2 Nonzero singular values from small matrices

Assume first that $L$ is positive definite. Let

$$
L=CC^\top,
\qquad Q=C^{-1}U,
\qquad QQ^\top=I.
$$

Then

$$
T=(VC)Q.
$$

Because $Q$ has orthonormal rows, the nonzero singular values of $T$ equal those of $VC$. Therefore the squared singular values are the eigenvalues of

$$
S=C^\top RC.
\tag{8}
$$

Equivalently one may use $L^{1/2}RL^{1/2}$. The product $LR$ has the same nonzero spectrum, but it need not be symmetric in the raw Euclidean coordinates. Prefer a symmetric positive-semidefinite eigensolve for numerical work. [DERIVATION]

### 8.3 Native encoder, decoder, and projector

Let $Z_r$ contain the leading eigenvectors of $S$. Define

$$
D_r=Z_r^\top C^{-1},
\qquad B_r=CZ_r,
\qquad P_r=B_rD_r.
\tag{9}
$$

Here $D_r$ encodes the hidden state into retained coordinates and $B_r$ decodes it into the native hidden space. Then

$$
D_rB_r=I_r,
\qquad P_r^2=P_r.
$$

In general $P_r\ne P_r^\top$. It is orthogonal in canonical coordinates but oblique in the original coordinates. The rank-$r$ compressed cut is

$$
T_r=VB_rD_rU=VP_rU.
$$

Its squared coefficient error is

$$
\|T-T_r\|_F^2=\sum_{j>r}\lambda_j.
\tag{10}
$$

This realizes the optimal rank-$r$ SVD approximation of this one matricization, not the globally optimal simultaneous low-rank approximation of every tensor-network bond.

### 8.4 Gauge transport

For an invertible hidden-coordinate change,

$$
U'=MU,\qquad V'=VM^{-1},
$$

we have

$$
L'=MLM^\top,
\qquad
R'=M^{-\top}RM^{-1},
$$

and

$$
L'R'=M(LR)M^{-1}.
\tag{11}
$$

Thus the spectrum is invariant. If $C'$ is another square-root factor of $L'$, then $C'=MCO$ for an orthogonal $O$. The canonical operators are orthogonally related, and consistently matched modes satisfy

$$
B_r'=MB_r,
\qquad
D_r'=D_rM^{-1},
\qquad
P_r'=MP_rM^{-1}.
\tag{12}
$$

Individual vectors are ambiguous within degenerate eigenspaces. Compare functional projectors or matched subspaces, not arbitrary eigenvector ordering or signs.

### 8.5 Rank-deficient upstream maps

If $L$ is singular, choose $C\in\mathbb R^{c\times s}$ with full column rank on the reachable support. Write $U=CQ$, $QQ^\top=I_s$, and diagonalize $C^\top RC$.

A left inverse of $C$ gives a native encoder on its range. An ordinary Euclidean pseudoinverse does not transform covariantly off that range under an arbitrary non-orthogonal $M$. Therefore:

- the retained functional maps and their action on reachable features are invariant;
- an extension of the projector to unreachable native directions is not uniquely specified without an additional convention;
- no theorem should claim full off-support projector covariance from a pseudoinverse alone.

The companion check `singular_support_transport` demonstrates this distinction.

### 8.6 Relation to original ODT

After Dooms' bottom-up canonicalization, the upstream map on a valid tree cut has $L=I$. Equation (8) reduces to the output-side Gram $S=R$, exactly the situation exploited by ODT.

Accordingly:

> The two-sided expression is a coordinate-explicit formulation of the same valid-cut singular-spectrum problem; it is not a substitute for proving that a proposed MACE interface is such a cut.

### 8.7 Why this does not solve an arbitrary graph

If removing one tensor-network edge does not separate the relevant external legs into two independent halves, the environment need not factor into $L$ and $R$. A general single-edge insertion can produce a four-index quadratic form

$$
\|\Theta-\Theta(P)\|^2
=\sum_{a,b,c,d}
(I-P)_{ab}\,
\mathcal E_{ab,cd}\,
(I-P)_{cd},
\tag{13}
$$

with no factorization $\mathcal E=L\otimes R$.

If the same logical projector is inserted at several tied occurrences, $\Theta(P)$ is generally polynomial in $P$, not even linear. The existence of small upstream and downstream Grams must be established for each construction; it cannot be assumed from the word “global.”

---

<a id="section-9"></a>

## 9. Primary algorithm: exact mixed-order decomposition of the first energy branch

### 9.1 Why start here

The first branch is nonlinear as a function of its aggregate features, but it has bounded correlation order and an explicit scalar coefficient representation. It avoids the deep global-leaf symmetrization problem and supplies direct body-order interpretation. It also permits nontrivial subspaces **before** products, unlike the trivial rank-one cut immediately before a scalar linear readout. [H:appA; H:expansion]

### 9.2 Canonical coefficient tensors

After the radial orthogonalization in (4), collect the canonical primitive functions into $q_z(t)$, including angular indices. Let

$$
b_z(X)=\sum_{t_j\in X}q_z(t_j).
$$

Compose the branch into

$$
E_{L,z}(X)
=c_z+\sum_{\nu=1}^{3}
K_{z,\nu}[b_z(X)^{\otimes\nu}].
\tag{14}
$$

The tensors $K_{z,\nu}$ include the radial maps' non-isometric factors, all relevant CG couplings, learned path weights, post-product linear maps, and the scalar readout. They are not isolated layer weights.

Store them as symmetry-aware factorizations rather than dense arrays whenever possible. Their dense versions are required only for small test cases.

### 9.3 Single-slot reduced operators

For a tensor $K_{z,\nu}$, define the reduced operator on slot $a$ by

$$
\rho_{z,\nu,a}
=
\operatorname{Tr}_{\text{all slots except }a}
\bigl(|K_{z,\nu}\rangle\langle K_{z,\nu}|\bigr).
\tag{15}
$$

It contains every coefficient of the branch at that order, including the downstream readout, while integrating out the other orthonormal primitive slots.

For a fully symmetric $K_{z,\nu}$, the single-slot operators agree after identifying their spaces. Retaining a common primitive subspace, however, changes **all** slots. The next step accounts for that explicitly.

### 9.4 A tied, multi-order score

Define

$$
\Gamma_z
=
\sum_{\nu=1}^{3}\beta_\nu
\sum_{a=1}^{\nu}\rho_{z,\nu,a}.
\tag{16}
$$

For fully symmetric tensors this is

$$
\Gamma_z=\sum_{\nu=1}^{3}\nu\beta_\nu\rho_{z,\nu,1}.
$$

The factor $\nu$ counts the number of projected occurrences. Omitting it would optimize a different weighting convention.

Choose an orthogonal projector $\Pi_z$ in canonical function coordinates. The reduced coefficients are

$$
K_{z,\nu}^{\Pi}
=\Pi_z^{\otimes\nu}K_{z,\nu}.
\tag{17}
$$

The same projector acts on every repeated slot, preserving permutation symmetry and the interpretation as one reduced aggregate representation.

### 9.5 The exact guarantee available here

Define the actual coefficient approximation error

$$
\mathcal E_z(\Pi)
=
\sum_{\nu=1}^{3}\beta_\nu
\|K_{z,\nu}-\Pi^{\otimes\nu}K_{z,\nu}\|_F^2.
\tag{18}
$$

For orthogonal $\Pi$,

$$
\boxed{
\mathcal E_z(\Pi)
\le
\operatorname{Tr}\bigl[(I-\Pi)\Gamma_z\bigr].
}
\tag{19}
$$

**Proof.** Let $\Pi_a$ denote $\Pi$ acting on slot $a$. These are commuting orthogonal projectors on different tensor factors. In their joint eigenbasis,

$$
I-\prod_{a=1}^{\nu}\Pi_a
\preceq
\sum_{a=1}^{\nu}(I-\Pi_a).
$$

For an orthogonal projector, the squared residual norm equals the associated quadratic form. Apply this inequality to $K_{z,\nu}$, then sum over orders with positive $\beta_\nu$. Each single-slot residual equals $\operatorname{Tr}[(I-\Pi)\rho_{z,\nu,a}]$. This gives (19). [DERIVATION]

For a fixed rank $r$, the leading eigenspace of $\Gamma_z$ minimizes the right-hand side. Consequently,

$$
\mathcal E_z(\Pi_r)
\le\sum_{j>r}\gamma_{z,j},
\tag{20}
$$

where $\gamma_{z,j}$ are the eigenvalues of $\Gamma_z$.

This is an explicit **upper-bound-optimal tied projection**, not a claim that an eigendecomposition globally minimizes the coupled objective in (18).

### 9.6 A mixed-order quasioptimality statement

Let $\Pi_*$ minimize (18) among orthogonal projectors with the same rank constraints. Each single-slot residual is at most the full product-projection residual, because the product subspace is contained in each single-slot subspace. Therefore,

$$
\operatorname{Tr}[(I-\Pi_*)\Gamma_z]
\le \nu_{\max}\,\mathcal E_z(\Pi_*).
$$

Combining this with (19) and the upper-bound optimality of $\Pi_r$ gives

$$
\boxed{
\mathcal E_z(\Pi_r)
\le
\nu_{\max}\,\mathcal E_z(\Pi_*),
\qquad \nu_{\max}=3.
}
\tag{21}
$$

In the corresponding norm, the approximation factor is $\sqrt3$. This is a specialization of the familiar projection logic behind higher-order SVD bounds, not a claim of a fundamentally new decomposition theorem. It applies to the stated orthogonal, shared-slot, mixed-order coefficient problem. It does not transfer unchanged to the physical force error or an arbitrary tied computation graph. [DERIVATION; related background T:tree]

### 9.7 Equivariant restriction

Let the primitive space decompose as

$$
\mathcal H_z
=\bigoplus_{\alpha=(\ell,p)}
\mathbb R^{c_{z,\alpha}}\otimes V_\alpha,
\qquad d_\alpha=2\ell+1.
$$

For a scalar invariant kernel and a rotationally invariant product measure, each reduced operator commutes with the group action. Hence

$$
\Gamma_z
=\bigoplus_\alpha
\Gamma_{z,\alpha}^{\mathrm{mult}}\otimes I_{d_\alpha}.
\tag{22}
$$

Use

$$
\Pi_z
=\bigoplus_\alpha
\Pi_{z,\alpha}^{\mathrm{mult}}\otimes I_{d_\alpha}.
$$

Compute the reduced multiplicity operator by tracing the magnetic indices and dividing by $d_\alpha$, provided the full block is of the stated form. The discarded trace contribution is

$$
\sum_\alpha d_\alpha
\sum_{j>r_{z,\alpha}}\gamma_{z,\alpha,j}.
\tag{23}
$$

Keep complete irreps. Never retain only selected magnetic components because their numerical eigenvalues appear small. If the compiled operator violates the block structure beyond numerical error, diagnose the compiler or measure before proceeding.

For a fixed rank in each multiplicity block, the leading block eigenspaces minimize the marginal bound. For a total coordinate budget $\sum_\alpha d_\alpha r_\alpha$, allocating ranks is a discrete constrained optimization problem; whole-multiplet costs must be respected. A small dynamic program can optimize an additive coordinate budget. Actual inference costs can couple several ranks nonlinearly and require a separate allocation search. The factor-three quasioptimality statement requires minimizing the marginal bound over the same admissible class used by the comparator; it is not automatically inherited by a heuristic budget allocation.

### 9.8 Relation to Dooms' algorithm

This construction retains the core ODT principle: orthogonalize the upstream function map, contract a composed coefficient environment, and truncate subspaces selected by that environment.

It is a **first-branch, mixed-order, equivariant specialization of ODT/HSVD ideas**, rather than verbatim Algorithms 1–3 on a deep binary χ-net. For a single binary core with equivalent slots, $\Gamma=2\beta\rho$, so it selects the same single-slot eigenspace as the corresponding ODT environment; the scalar factor only changes the score normalization.

The shared-slot projection and explicit bound make the proposed first milestone mathematically self-contained. A deep-MACE implementation must earn its additional claims separately.

### 9.9 Optional refinement after spectral initialization

Within the retained rank constraints, minimize the exact weight-only coefficient objective (18), using the eigenspaces of $\Gamma$ as initialization. This is an optional symmetric Tucker-style subspace refinement, not a new ODT identity.

Keep the original spectral result and the refined result as separate methods. Report optimization cost, initialization sensitivity, final objective, and gauge equivariance. If an optimizer uses derivatives, they are derivatives of a deterministic coefficient objective, not task-data gradients; do not describe that arm as “no gradients” without qualification.

### 9.10 Cross-species aggregation requires additional care

If each central species has a different upstream metric $L_z$, simply forming

$$
\left(\sum_z L_z\right)
\left(\sum_z R_z\right)
$$

introduces cross-species terms and does not generally equal the correct shared-projector objective.

Primary results therefore use $\Pi_z$ separately, or a common **physical primitive basis** in which every $K_{z,\nu}$ is represented and then aggregate the reduced operators there. The latter common basis may have dimension greater than the native channel width; its rank must not be mislabeled as a native channel rank.

A single native shared projector is an additional constrained optimization problem. Evaluate it using its exact multi-species objective rather than asserting that averaged two-sided matrices solve it.

---

<a id="section-10"></a>

## 10. Efficient contractions without dense global tensors

### 10.1 Preserve the existing factorization

A schematic separable coefficient representation is

$$
K_\nu=\sum_{s=1}^{S_\nu}
\theta_s\,
C_s^{\mathrm{ang}}\otimes
\bigotimes_{a=1}^{\nu}r_{s,a},
\tag{24}
$$

where $C_s^{\mathrm{ang}}$ carries the allowed angular couplings and each $r_{s,a}$ is a radial/species coefficient vector in an orthonormal primitive space. The precise term indexing is derived from the checkpoint; it is not assumed to be a generic unrestricted CP decomposition.

Keep channel maps as transformations of these factors. Applying a dense channel basis change does not require materializing the complete $c^\nu$ coefficient tensor if the transformed factors remain explicitly available.

### 10.2 A useful contraction identity

For a fully separable real tensor

$$
K=\sum_s\theta_s\bigotimes_a r_{s,a},
$$

its reduced operator on slot $a$ is

$$
\rho_a
=
\sum_{s,t}\theta_s\theta_t
\left[\prod_{b\ne a}
\langle r_{s,b},r_{t,b}\rangle\right]
\,r_{s,a}r_{t,a}^\top.
\tag{25}
$$

With angular tensors, include the corresponding exact angular contraction and matching of irrep labels. This equation exposes a path to pairwise-factor contractions rather than expansion of every coefficient. [DERIVATION]

Cache radial overlaps, angular contraction tables, and repeated species/path structures. Use matrix-free $\rho v$ products when the retained multiplicity space is large. An approximation such as low-rank compression of overlap matrices must have its own error record.

### 10.3 Cross terms are essential

When several coupling paths contribute to the same coefficient tensor, form the sum before taking its norm or include all path-pair cross terms explicitly. Replacing

$$
\left\|\sum_s K_s\right\|^2
$$

with

$$
\sum_s\|K_s\|^2
$$

discards interference and cancellation. It can turn an exact null direction into an apparently important one, defeating the HTML's central nullspace lesson.

Likewise, degrees are kept in a declared direct sum only because that is the chosen coefficient norm. Within one degree, artificial branch labels must not make cancelling terms orthogonal.

### 10.4 Do not allocate Kronecker powers

The notation $C^{\otimes\nu}$ specifies a multilinear transformation. Implement it as successive tensor contractions, sparse path transformations, or transformations of the factors in (24).

For $c=96$, a dense scalar cubic coefficient array has $96^3=884,736$ entries, about 7.1 MB in float64. A dense channel-to-channel cubic map has $96^4=84,934,656$ entries, about 679 MB. These are channel-only counts, not estimates for the full angular and species-expanded MACE computation. Angular indices, path multiplicities, species, and multiple tensors can dominate memory.

A genuinely dense $(\nu+1)$-leg core of width $c$ requires $O(c^{\nu+1})$ storage, and an output-by-input QR/RQ can cost $O(c^{\nu+2})$. Dooms' quartic bound is specific to binary cores and the repeated tree structure. [DERIVATION; D:AF]

### 10.5 Memory preflight

Before any contraction, record:

- the exact einsum or tensor-network expression;
- each leg's dimension and meaning;
- contraction path and peak intermediate size;
- estimated floating-point work;
- whether any approximation is inserted;
- whether equivalent computations are shared or duplicated;
- a hard memory limit that rejects unsafe materialization.

Test a tiny instance densely, compare it with the factorized implementation, and only then scale the same expression.

---

<a id="section-11"></a>

## 11. Quotienting architectural redundancy correctly

### 11.1 Construct the path map from the actual CG convention

Flatten each fully symmetrized coupling tensor into a column of a matrix $T_{\mathrm{path}}$. Include any path normalization factors used by the checkpoint. Compute its supported row space using a deterministic rank-revealing factorization.

If $Q_{\mathrm{path}}$ has orthonormal rows spanning that row space,

$$
\omega=Q_{\mathrm{path}}W,
\qquad
T_{\mathrm{path}}W
=T_{\mathrm{path}}Q_{\mathrm{path}}^\top\omega.
\tag{26}
$$

This reproduces the HTML's reduced-coordinate construction, with a distinct symbol to avoid confusing it with radial curves. [H:appC-reduced, Eq.(19)]

### 11.2 Required tests

Verify the reported path ranks only when the checkpoint's angular cutoff, parity, output irrep, correlation order, and CG basis match the source. Recompute rather than hard-code $23\to8$.

For each tested cell:

- sample or construct a vector in $\ker T_{\mathrm{path}}$;
- add it to the stored path coefficients;
- check the composed symmetric tensor, polynomial evaluation, branch energy, and forces;
- perturb a retained coordinate as a positive control;
- convert to the reduced basis, reload, and compare again.

Testing the coefficient identity is data-free. Geometric samples are verification inputs, not discovery data.

### 11.3 What an ODT-like test can and cannot recover

A decomposition of the path-to-symmetric-tensor map can recover its 15-dimensional kernel. A decomposition of one fixed model's channel interface is a different object and need not contain fifteen zero eigenvalues.

After composing the quotient path coefficients into $K_\nu$, adding a known path-null vector should leave $K_\nu$, $\Gamma$, and the recovered functional subspaces unchanged. This is the appropriate integration test between the HTML's quotient and the global decomposition.

### 11.4 Residual compactness is not automatically learned

Further rank restrictions may follow from scalar outputs, the last radial MLP width, symmetry selection rules, exact radial dependencies, channel-factorized parameterization, or zero branches. Before labeling additional small eigenvalues as learned global redundancy, compare with these ceilings and with initialization-matched controls.

Use the phrase **checkpoint-specific compactness beyond the explicitly removed nullspace** until the architecture-versus-learning distinction has been tested.

### 11.5 Fine-tuning is outside the primary construction

The primary method does not retrain the potential. If optional fine-tuning follows compression, report pre- and post-fine-tuning results separately.

The HTML's Appendix D explains that reducing redundant coordinates changes adaptive-optimizer dynamics and reports experiments about that difference. Exact forward equivalence is not optimizer-trajectory equivalence. Any effort to reproduce those dynamics requires matching the optimizer variant, moments, weight decay, normalization conventions, and parameterization. [H:appD]

---

<a id="section-12"></a>

## 12. From canonical modes to invariant physical objects

### 12.1 Export a feature map, not just an eigenvector

An eigenvector of a canonical reduced operator is expressed in a particular orthonormal coordinate system. The object to inspect is the function it defines, together with its use by the rest of the computation.

For a first-branch mode, let the canonical eigenvector be $z_k$. Its native encoder covector and decoder vector are

$$
 d_k=C^{-\top}z_k,\qquad b_k=Cz_k.
\tag{27}
$$

On the supported space, the scalar radial image is

$$
 \phi_k(z,z',r)=d_k^\top F_{zz'}^{(\ell)}(r)
 =z_k^\top q_{zz'}^{(\ell)}(r).
\tag{28}
$$

For a non-scalar irrep, combine this radial image with the corresponding spherical-harmonic components. Retain the entire irrep multiplet, rather than interpreting a chosen magnetic component as an invariant scalar.

Equation (28) is the direct use of the HTML's image construction. The additional contribution is the source of the covector: it is selected by a downstream-composed coefficient operator rather than supplied by an SAE, a benchmark search, or a raw local singular vector. [H:curves, Eqs.(7)–(8); DERIVATION]

Under an admissible coordinate change, $d_k'=M^{-\top}d_k$, $b_k'=Mb_k$, and $F'=MF$. Consequently the image and encoded feature remain unchanged. Native decoder coordinates need not remain unchanged, and they should not.

### 12.2 The deep case needs a different semantic map

For a deeper bond, the upstream object may already be a polynomial of many neighbors and multiple message-passing steps. Its mode is then

$$
 \phi_{b,k}(X)=d_{b,k}^{\top}h_b(X),
$$

or the corresponding function of independent formal input slots in the chosen lift. It cannot generally be represented by one first-layer radial curve.

A deep semantic renderer must propagate the encoder through the actual upstream polynomial program and produce a many-body or receptive-field kernel. Mapping an arbitrary deep eigenvector directly through the first-layer matrix $F$ would be a type error unless an explicit transport map has been derived.

### 12.3 An intervention requires both encoder and decoder

For a retained set of modes $S$, use

$$
 P_S=\sum_{k\in S}b_kd_k^\top.
\tag{29}
$$

A one-mode removal is $I-b_kd_k^\top$, with $d_k^\top b_k=1$. This specifies both what is measured and what is removed. It is not generally $I-d_kd_k^\top$.

This distinction operationalizes the HTML's removal-rule finding. An orthogonal projection after whitening becomes a particular oblique projection in the native channels. Other native orthogonal or covariance-derived oblique edits are legitimate experimental comparators, but they are different interventions. [H:comparing, Eq.(9); H:calibrations]

At an actual MACE interface, derive the weight edit from the interface registry. For a producer $A$ and consumer $B_z$, a retained bottleneck can be implemented as $D_SA$ followed by $B_zB_S$ where shapes permit. At a polynomial consumer, apply the appropriate decoder to **every** affected input slot. Do not infer the side of multiplication from the symbol $B_z$ alone; the HTML uses different layer descriptions that must be reconciled with the exact computation being edited.

### 12.4 Interpret the interaction tensor as well as the modes

After reduction,

$$
 E_{L,z}(X)=c_z+
 \sum_{\nu=1}^{3}\sum_{k_1,\ldots,k_\nu}
 \widehat K_{z,\nu;k_1\ldots k_\nu}
 \prod_{a=1}^{\nu}\phi_{k_a}(X).
\tag{30}
$$

Even if every $\phi_k$ has a simple radial image, off-diagonal entries of $\widehat K$ couple the modes. A feature associated with short-range carbon–oxygen geometry might matter only through an interaction with another mode.

There is no automatic, unique decomposition $E=\sum_k E_k$. Possible conventions include splitting mixed terms among their indices, reporting the full interaction core, or using nested deletion increments. The first and third introduce attribution conventions; nested increments also depend on removal order. The primary report should show the reduced interaction core and joint interventions before assigning individual energy contributions.

This follows the spirit of Dooms' interpretation of diagonal and off-diagonal core entries, but does not assume that MACE will exhibit the same near-linear early-layer structure. [D:S4]

### 12.5 Required mode or subspace card

Each reported mode, degenerate block, or selected subspace must have a machine-readable record containing:

| Field | Required content |
|---|---|
| Identity | Checkpoint hash, interface ID, central-element conditioning, irrep and parity, source tensor target. |
| Metric | Radial interval, species weights, angular normalization, order weights, and output metric. |
| Spectral statistic | Eigenvalue, normalization, rank criterion, spectral gap, and whether the statistic is a cut singular value or an aggregate marginal score. |
| Maps | Canonical basis, native encoder, native decoder, supported-space convention, projector. |
| Functional image | Element-pair radial functions or explicit deeper kernels, with evaluation and integration accuracy. |
| Composition | Dominant reduced-core interactions and body-order content, including mixed-mode terms. |
| Stability | Gauge tests, quadrature sensitivity, metric sensitivity, seed/model comparison where available. |
| Intervention | Exact weight edit, all consumers affected, edit norm, dual, and branch-only versus shared-model scope. |
| Fidelity | Energy, relative-energy, force, and stress results when relevant. |
| Interpretation | Proposed semantic description, supporting evidence, matched controls, and unresolved alternatives. |

“Globally important” must always name the target and norm. For example: **important for the first linear-energy branch under the declared independent-neighbor coefficient norm**.

### 12.6 Degeneracy and cross-model comparison

An isolated eigenvector is stable only when the relevant spectral separation is large compared with numerical or modeling perturbations. Near-degenerate eigenvalues justify a subspace, not an individually identifiable direction.

Compare subspaces in a common function space using principal angles, projector overlap, or their action on an agreed basis. Align signs only after specifying a convention. Do not compare raw channel cosines across independently trained checkpoints as a test of shared mechanisms.

Across models with different radial spans, build a common evaluation or integration space containing both spans and document its approximation error. Do not pretend that equal channel index or equal ordinal eigenvalue establishes correspondence. The HTML's limited cross-model curve agreement is a reason to test this, not a reason to assume either universal alignment or universal non-alignment. [H:appG-curves]

---

<a id="section-13"></a>

## 13. The body-order bridge: exact formulas and limits of attribution

### 13.1 Correlation order is not the same as distinct-neighbor count

Write the symmetric kernel evaluation as

$$
 k_{z,\nu}(t_1,\ldots,t_\nu)
 =K_{z,\nu}[q_z(t_1)\otimes\cdots\otimes q_z(t_\nu)].
$$

Then the first branch expands as

$$
 E_{L,z}(X)-c_z
 =\sum_j k_{z,1}(t_j)
 +\sum_{j,k}k_{z,2}(t_j,t_k)
 +\sum_{j,k,l}k_{z,3}(t_j,t_k,t_l).
\tag{31}
$$

The sums include repeated neighbor indices. A cubic term can therefore contribute to a pair interaction, a three-body interaction, or a four-body interaction when classified by distinct atoms. It is incorrect to call every cubic correlation term exclusively four-body.

### 13.2 Explicit distinct-neighbor terms

Using the no-factorial convention of (31), the inclusion–exclusion terms are

$$
\begin{aligned}
 V_z(\{j\})={}&k_{z,1}(t_j)
 +k_{z,2}(t_j,t_j)
 +k_{z,3}(t_j,t_j,t_j),\\
 V_z(\{j,k\})={}&2k_{z,2}(t_j,t_k)
 +3k_{z,3}(t_j,t_j,t_k)
 +3k_{z,3}(t_j,t_k,t_k),\\
 V_z(\{j,k,l\})={}&6k_{z,3}(t_j,t_k,t_l)
 \quad\text{for distinct }j,k,l.
\end{aligned}
\tag{32}
$$

The central atom contributes one to body order, so these are two-, three-, and four-body site-energy terms. If the implementation includes $1/\nu!$ in its kernel definition, change the combinatorial factors consistently rather than mixing conventions.

The companion numerical checks verify these formulas against direct subset inclusion–exclusion on a synthetic cubic branch. [DERIVATION]

### 13.3 Recover the HTML's exact construction

For any site-energy function $e_z$, define

$$
 V_z(S)=\sum_{T\subseteq S}(-1)^{|S|-|T|}e_z(T),
 \qquad
 e_z(N)=\sum_{S\subseteq N}V_z(S).
\tag{33}
$$

This is the HTML's Möbius construction. To establish the first branch's vanishing beyond three neighbors, apply its subset lemma separately to each monomial: a monomial with neighbor support $U$ depends only on $T\cap U$, and its contribution to $V_z(S)$ vanishes unless $S\subseteq U$. Each monomial has at most three distinct neighbors; linearity completes the proof. The whole site energy need not depend on a single fixed three-neighbor set. [H:appG-expansion, Eqs.(25)–(26); DERIVATION]

For the first branch, two independent implementations should agree: the coefficient formulas (32), and the subset evaluator (33). This is a high-value integration test of the symbolic compiler, radial tabulation, symmetrization, and readout composition.

### 13.4 How ODT-selected subspaces enter the body-order analysis

Replace $K_{z,\nu}$ with its projected tensor and apply the same formulas. This yields:

- the original branch's body-order kernels;
- the retained subspace's kernels;
- the discarded difference kernels;
- changes caused by a specified individual or joint mode removal.

Because these are functions, they can be compared across native gauges. Because they are derived from complete coefficients, they retain cancellation between channel paths. Neither property alone makes them independent physical forces or unique mechanisms.

A useful research question is whether **the discarded functional error is simple enough to explain**, not just whether retained radial images look familiar.

### 13.5 Chemical case selection must respect branch scope

The HTML identifies a particularly useful positive and negative contrast. Its small-model ethane torsional barrier lies largely in the linear branch, whereas the difluoroethane gauche preference requires the nonlinear branch. The latter's small-model contributions have opposite signs: approximately +35.1 meV from the linear branch and −59.9 meV from the MLP branch, totaling −24.8 meV. The other released sizes also show opposite signs. [H:appG-conformers]

Accordingly:

**First-branch case study:** investigate whether a compact mode interaction accounts for the branch's contribution to the ethane barrier, then measure its full-model impact without claiming the branch contains all chemistry.

**Full-model challenge:** investigate the difluoroethane preference only after the nonlinear branch is included. A first-branch-only explanation of its total sign would contradict the source's readout split.

Pair-well locations and angular minima are additional descriptive tests, but the HTML shows that isolated-pair minima can resemble diatomic rather than bonded-molecule distances. They are not automatic validation of in-molecule chemical semantics. [H:terms]

### 13.6 Cancellation and extrapolative fragments

Report the HTML's cancellation diagnostic,

$$
 \chi_b(X)=
 \frac{\sum_{|S|=b-1}|V_z(S)|}
 {|\sum_{|S|=b-1}V_z(S)|},
\tag{34}
$$

with a denominator floor and a separate flag when the signed sum is near zero. Do not rank importance solely by the numerator.

The source reports large cancellations in nonlinear-readout subset terms, with individual differences up to approximately 1.2 eV and cancellation ratios of roughly 13–19 in the studied example. These are reported source measurements, not new results. [H:appG-conformers]

Subset fragments may be far outside the training distribution. The expansion is an identity for the model; an individual term is not thereby a validated quantum-chemical interaction. Supplementary body-order research likewise motivates keeping exact decomposition separate from physical attribution, within the datasets and architectures actually studied there. [H:expansion; B:body]

---

<a id="section-14"></a>

## 14. Error guarantees: what is bounded and what is not

### 14.1 Keep error spaces separate

The project involves several norms:

| Error | Meaning | Automatically controlled by a coefficient-spectrum tail? |
|---|---|---|
| Coefficient error | Distance between tensors in the declared external basis and metric. | Yes, for the proved single-cut or first-branch projection result. |
| Structural function error | Error under a declared geometric or neighbor measure. | Only if the coefficient metric represents that measure or a valid conversion bound is supplied. |
| Pointwise energy error | Error on one configuration. | Requires the input feature norm or evaluation-functional bound. |
| Force error | Error in coordinate derivatives. | Requires derivative bounds or a separately certified derivative norm. |
| Reference error | Difference from quantum-chemistry targets. | No. The decomposition approximates the model, which may itself be wrong. |
| Trajectory stability | Behavior over a dynamical rollout. | No. Small average one-step errors alone are insufficient. |

This distinction is essential to both sources: Dooms leaves loss guarantees to future work, while the HTML distinguishes representation, intervention effect, and chemical meaning. [D:S3; H:synthesis; H:limitations]

### 14.2 A pointwise first-branch energy bound

Let

$$
 \epsilon_{\mathrm{coef}}^2
 =\sum_{\nu=1}^{3}\beta_\nu\|\Delta K_{z,\nu}\|_F^2.
$$

For a fixed configuration with canonical aggregate $b=b_z(X)$, Cauchy–Schwarz gives

$$
 |\Delta E_{L,z}(X)|
 \le
 \epsilon_{\mathrm{coef}}
 \left(\sum_{\nu=1}^{3}
 \frac{\|b\|^{2\nu}}{\beta_\nu}\right)^{1/2}.
\tag{35}
$$

A constant-term error must be added separately. An exact, unmodified constant contributes no truncation error.

Equation (35) is a valid consequence of the coefficient norm. It can be loose when cancellation makes the actual energy small. It is not a relative-energy guarantee unless the denominator of the chosen relative quantity is also controlled. [DERIVATION]

### 14.3 A coordinate-gradient bound

Let $x$ denote the coordinates affecting the site and $J_b=\partial b/\partial x$. With fixed coefficients and a differentiable feature map,

$$
 \|\nabla_x\Delta E_{L,z}(X)\|
 \le
 \epsilon_{\mathrm{coef}}\,\|J_b\|_{\mathrm{op}}
 \left(\sum_{\nu=1}^{3}
 \frac{\nu^2\|b\|^{2\nu-2}}{\beta_\nu}\right)^{1/2}.
\tag{36}
$$

The derivative of a $\nu$-fold product contributes at most $\nu\|b\|^{\nu-1}\|J_b\|_{\mathrm{op}}$. Apply Cauchy–Schwarz across orders to obtain (36). For $\nu=1$, interpret $\|b\|^0=1$, including at $b=0$.

The physical force convention in this specification is $F=-\nabla_x E$; the error norm is unaffected by the sign. Uniform bounds require a specified coordination cap, radial domain, and control of the radial and angular derivatives. Angular derivatives may involve inverse distance, so a positive lower interatomic-distance bound or another regularity argument is necessary. Neighbor-list changes and cutoff smoothness must be handled explicitly.

For a total energy, sum site contributions with their actual coordinate dependencies. Per-atom energy normalization does not remove overlapping contributions to one atom's force. [DERIVATION]

### 14.4 Uniform domains and feature evaluation

For a chosen domain $\mathcal D$, a sufficient strategy is to certify

$$
 \sup_{X\in\mathcal D}\|b_z(X)\|\le B_z,
 \qquad
 \sup_{X\in\mathcal D}\|J_{b_z}(X)\|_{\mathrm{op}}\le J_z.
$$

A neighbor cap $N_{\max}$ and a bound on the one-neighbor feature norm can give $B_z\le N_{\max}\sup_t\|q_z(t)\|$. This is data-free but can be conservative. Its existence does not imply that a useful small rank will meet a practically meaningful force tolerance.

Report both certified bounds and measured errors. Do not label finite random testing as certification over a continuous domain.

### 14.5 Multi-bond tree truncation

For a genuine tree with an appropriate hierarchical projection construction, use a sum of discarded **squared singular values** over the actual projected tree occurrences. If one logical layer appears at many tree nodes, count those occurrences or prove a valid weighted aggregation.

Do not substitute the number of learned layers for the number of projections in the unfolded tree. Do not carry a bound through a refolded, shared MACE edit without showing that the refolded edit realizes the certified projected tensor.

The first-branch common-projector bound in Section 9 is proved separately and does not depend on importing the source's printed Gram-norm condition. [D:S3; T:tree; DERIVATION]

### 14.6 Numerical and modeling errors need compatible units

Keep separate records for radial quadrature, metric rank restriction, coefficient compilation, factorized contraction, spectral projection, and final evaluator error.

An energy bound may take the form

$$
 \sup_{\mathcal D}|E-E_{\mathrm{export}}|
 \le \epsilon_{\mathrm{representation},E}
 +\epsilon_{\mathrm{projection},E}
 +\epsilon_{\mathrm{export},E},
\tag{37}
$$

but only after each term is defined in the **same energy norm on the same domain**. A coefficient Frobenius error and an activation approximation error cannot simply be added as unnamed quantities. Force certificates require a parallel derivative-error budget.

### 14.7 Spectral stability

A small perturbation in the canonical reduced operator can move eigenvectors substantially when eigenvalues are nearly degenerate. Measure convergence of invariant subspaces and projectors, not merely of sorted eigenvalues. The error in a quadrature-derived metric can also be amplified by whitening ill-conditioned directions.

Rank selection near the numerical support boundary must include sensitivity to precision and integration order. Stable lower-dimensional blocks are preferable to overly precise claims about unstable individual modes.

---

<a id="section-15"></a>

## 15. Extending through both message-passing layers

### 15.1 An exact polynomial target is possible without a polynomial head

For an audited architecture whose message-passing operations are polynomial in the learned radial primitives and previous hidden features, the second-layer hidden map is a polynomial tensor program. The final head is a separate nonlinear map.

This does not mean the entire potential is a polynomial in Cartesian coordinates. It also does not mean that every current MACE variant has the required polynomial hidden computation. Feature-dependent nonlinear gates, normalizations, and architecture options must be checked. The official source contains multiple interaction-block variants, so the checkpoint's actual module graph is authoritative for implementation. [H:appA; M:code]

Useful exact polynomial observables include the inputs to the last head, its linear preactivations, and the first energy branch. The first readout remains a first-layer branch; appending its scalar weights to the end of a different hidden map would define a new observable rather than recover the original readout.

### 15.2 Receptive fields, topology, and variable size

A model operating on arbitrary atom sets is not one finite tensor with an unspecified number of input legs. The specification must choose one of the following:

**A symbolic family of local environments.** Use a bounded receptive-field and neighbor-pattern representation, with formal sums over admissible patterns.

**A fixed-size padded family.** Specify maximum atom or neighbor counts, species variables, masks, and the meaning of absent neighbors.

**A fixed graph topology.** Analyze a topology-conditioned operator. This is not an architecture-wide result unless a family of graphs is covered.

A graph chosen from an actual molecule is an input-dependent structural choice, even if no activations are fitted. Label such an analysis as topology-conditioned rather than claiming it is universally checkpoint-only.

Shared physical atoms and edges create relations between the primitive inputs of different branches. A norm treating duplicated primitive slots independently is a formal coefficient norm; a physical function norm must enforce their correlations and geometric consistency.

### 15.3 Tree unfolding is an option, not a free complexity guarantee

An acyclic computation graph can often be unfolded by duplicating shared subcomputations. This may create a tree to which hierarchical methods apply. It can also multiply the number of occurrences, destroy cheap weight-sharing opportunities, and produce a lift whose independent slots no longer represent physically independent variables.

Conversely, preserving a compact shared graph can leave a contraction problem whose separator requires several indices or a higher-order environment. Low parameter count alone does not guarantee a low-width contraction path.

The project must measure this tradeoff. Neither “MACE contains graph cycles, so ODT is impossible” nor “MACE is polynomial, so the original recursion works unchanged” is an adequate argument.

### 15.4 Tied projectors across occurrences

Let a logical channel space occur at positions $a=1,\ldots,m$ in an unfolded computation. Independent cut projectors $P_a$ may give a useful tensor approximation but need not refold into one shared MACE channel reduction.

A native-compatible objective is instead

$$
 J(P)=\|\Theta-\Theta(P,\ldots,P)\|^2.
\tag{38}
$$

This is generally a nonlinear optimization problem in $P$. Summing original single-occurrence environments can be a spectral initialization or a provable upper-bound method in special commuting-product settings, as in Section 9. It is not universally the exact optimum of (38).

Record whether a result is:

- an independent-occurrence tensor approximation;
- a tied, common-subspace approximation with a proved bound;
- a coefficient-objective optimized tied approximation;
- a heuristic tied approximation validated empirically.

These are different scientific claims.

### 15.5 Residual and sum branches

Residual addition may be represented through direct sums, augmented constants, or explicit sum nodes. The selected representation affects ranks unless equivalent terms are recombined in the same external function space.

For example, representing two cancelling branches as orthogonal labeled outputs would make them appear large even though their sum is zero. The target must contract their sum before computing the function norm when the real observable is their sum.

Mixed correlation orders are deliberately separated in Section 9's declared direct-sum norm; that choice must not be confused with making arbitrary computational paths orthogonal.

### 15.6 Milestone for the two-layer extension

A two-layer extension is considered algorithmically complete only after it provides:

1. a formal input and output space with explicit metrics and graph/domain scope;
2. a reproducible map from the audited checkpoint to the tensor program;
3. a dense small-instance oracle agreeing with the original polynomial observables;
4. an efficient contraction strategy with recorded peak memory;
5. a precisely defined tied or untied truncation scheme;
6. a correctly scoped error statement;
7. an exported evaluator reproducing the projected target.

Only then should its spectra be interpreted or compared with a first-branch method.

---

<a id="section-16"></a>

## 16. Two distinct routes to the original nonlinear head

### 16.1 Route A: keep the head exact and decompose its polynomial inputs

Suppose the audited head has the form

$$
 E_N(X)=\sum_{j=1}^{m}a_j\rho_j(s_j(X))+c,
 \qquad s(X)=Wh^{(2)}(X)+b.
\tag{39}
$$

The source HTML describes a one-hidden-layer head with sixteen units for the studied models. Confirm its actual width, activation, masks, biases, and scaling in the loaded checkpoint. The official `NonLinearReadoutBlock` confirms the general linear–activation–linear structure, but the current source tree is not a pinned historical checkpoint implementation. [H:appA; M:code]

Construct an exact polynomial representation of the preactivations $s_j$ and retain the original activations in the exported evaluator. This avoids polynomializing the head itself.

A head-aware coefficient objective can give different weights to different $s_j$, using the output coefficients and derivative bounds. This is an observable-preservation objective, not the exact SVD of the complete nonlinear energy function.

### 16.2 Energy and force propagation through the exact head

Let $\delta s_j=s_j-s_j'$. On a domain containing both original and compressed preactivations, suppose

$$
 |\rho_j'|\le L_{1,j},\qquad |\rho_j''|\le L_{2,j}.
$$

Then

$$
 |E_N-E_N'|
 \le\sum_j |a_j|L_{1,j}|\delta s_j|.
\tag{40}
$$

A useful consequence is

$$
 |E_N-E_N'|
 \le
 \left(\sum_j |a_j|L_{1,j}\right)^{1/2}
 \left(\sum_j |a_j|L_{1,j}|\delta s_j|^2\right)^{1/2}.
\tag{41}
$$

For differentiable activations with the stated second-derivative bound,

$$
 \|\nabla E_N-\nabla E_N'\|
 \le\sum_j |a_j|
 \left[
 L_{1,j}\|\nabla\delta s_j\|
 +L_{2,j}|\delta s_j|\,\|\nabla s_j'\|
 \right].
\tag{42}
$$

These are elementary composition bounds. They can be loose; they justify a head-aware surrogate target but do not show that its spectrum predicts the best nonlinear-energy truncation. Nondifferentiable activations or different head structures require their own assumptions and bounds. [DERIVATION]

### 16.3 Avoid the terminal bottleneck trap

The linear map into an $m$-unit head has rank at most $m$. Rediscovering this at that exact interface is architecture-forced and not evidence for a small number of deep chemical mechanisms.

A nontrivial result must reduce a larger earlier interface while preserving the head's relevant observables or full energy. It should outperform the obvious strategy of merely absorbing a terminal linear map into its predecessor.

### 16.4 Route B: build a polynomial shadow of the head

Choose a compact interval for each preactivation and approximate

$$
 \rho_j(s)\approx p_{j,K}(s)
 =\sum_{q=0}^{K}c_{j,q}s^q.
\tag{43}
$$

The shadow is a new analysis representation. It can be generated deterministically from the head weights, activation, and declared domain. Its coefficient graph may use $q$-fold products or factored polynomial evaluation instead of dense tensor powers.

Required approximation records are

$$
 \sup_{s\in I_j}|\rho_j(s)-p_{j,K}(s)|\le\epsilon_{0,j},
 \qquad
 \sup_{s\in I_j}|\rho_j'(s)-p_{j,K}'(s)|\le\epsilon_{1,j}.
\tag{44}
$$

An error certificate for scalar values alone is insufficient for forces. A Taylor expansion without a remainder bound on the declared interval is not a certificate.

The interval must contain both the original and the altered network's preactivations over the claimed configuration domain. A bound based only on the original model can cease to apply after a large channel projection.

### 16.5 Polynomial degree is not the only computational issue

Polynomializing the head can duplicate the entire polynomial backbone many times. Maintain factorizations and explicit sharing where possible, but measure contraction complexity rather than assuming the head's small width solves it.

Approximation degree, representation choice, whitening, and low-rank truncation interact. Pointwise convergence of shadow energies as $K$ increases does not automatically imply convergence of coefficient Frobenius norms or individual spectral modes. Compare functions and stable subspaces across degrees under compatible external metrics.

### 16.6 Mapping a shadow truncation back to the original model

There are two different outputs:

**Shadow evaluator:** a polynomial approximation with its own truncated tensor representation.

**Projected original evaluator:** the original nonlinear activation retained, with specified native channel projectors inserted or absorbed into its weights.

The second requires showing that the shadow's selected maps correspond to well-defined native interfaces and that the induced original-model modification preserves the relevant preactivation domain. Evaluate both outputs separately. Agreement of one does not establish agreement of the other.

### 16.7 Recommended priority

Use Route A before Route B when polynomial preactivation compilation is tractable. It preserves the exact original activation and separates the difficult graph-contraction problem from activation approximation. Route B is valuable for a full coefficient-tensor research question, but it introduces a further modeling choice and error budget.

Neither route changes the primary discovery requirement: no activation dataset is used to fit the subspace in the principal weight-derived arm. Data-assisted variants may be useful baselines and must be labeled as such.

---

<a id="section-17"></a>

## 17. Implementation architecture and interfaces

### 17.1 Repository layout

The implementation should be organized around auditable objects rather than one monolithic tensor compiler.

```text
mace_odt/
  provenance/
    source_registry.py        # Source locators and immutable experiment metadata.
    checkpoint_manifest.py    # Hashes, versions, configuration, module graph.
  architecture/
    inspect.py                # Read-only module inventory and interface registry.
    adapters.py               # Explicit supported-checkpoint adapters.
    graph.py                  # Typed tensor-program graph with leg semantics.
  quotient/
    cg_paths.py               # Symmetrized path-map construction.
    exact_reduction.py        # Supported coordinates and conversion.
  functions/
    radial.py                 # Exact learned radial evaluator and its derivatives.
    measures.py               # Species/radial/angular measure definitions.
    quadrature.py             # Controlled overlap integration.
    canonical.py              # Supported function-space orthogonalization.
  coefficients/
    first_branch.py           # Complete first-readout coefficient construction.
    factorized.py             # CP/path/CG-aware storage and contractions.
    dense_oracle.py           # Tiny explicit coefficient tensors only.
    body_order.py             # Density expansion and inclusion–exclusion.
  decomposition/
    chi_reference.py          # Original binary χ-net RQ/EVD/truncation reference.
    bridge.py                 # Valid-cut two-sided decomposition.
    mixed_order.py            # Γ construction and common-subspace projection.
    equivariant.py            # Irrep multiplicities, parity, rank allocation.
    tied_graph.py             # Explicit experimental extensions, not default.
  export/
    branch_evaluator.py       # Branch-only exact/projected evaluator.
    native_bottleneck.py       # Legal fixed encoder/decoder insertions.
    compressed_evaluator.py   # Optional optimized compiled evaluation path.
  semantics/
    mode_cards.py
    radial_images.py
    interaction_cores.py
    subspace_alignment.py
  evaluation/
    fidelity.py
    intervention_controls.py
    chemical_cases.py
    runtime.py
    statistics.py
  tests/
    test_bridge.py
    test_chi_reference.py
    test_quotient.py
    test_radial_rewrite.py
    test_first_branch.py
    test_gauge_transport.py
    test_body_order.py
    test_force_bounds.py
    test_export.py
  experiments/
    configs/
    run_discovery.py
    run_validation.py
    run_test.py
    aggregate.py
```

The tree compiler, mixed-order decomposition, and nonlinear-head extensions should expose separate method names. A common API is useful; a common name that conceals different assumptions is not.

### 17.2 Minimum typed objects

```python
@dataclass(frozen=True)
class InterfaceSpec:
    identifier: str
    producer_path: str
    consumer_paths: tuple[str, ...]
    tensor_axes: tuple[str, ...]
    multiplicities: dict[str, int]  # Keys encode both ell and parity.
    central_species_conditioned: bool
    tied_occurrence_group: str | None
    gauge_scope: str
    target_scope: str

@dataclass(frozen=True)
class MeasureSpec:
    radial_interval: tuple[float, float]
    radial_units: str
    radial_weight_name: str
    species_weights: dict[str, float]
    angular_normalization: str
    order_weights: dict[int, float]
    quadrature_order: int

@dataclass
class DecompositionResult:
    target_id: str
    interface_id: str
    measure_id: str
    method: str
    eigenvalues: Array
    encoder: Array
    decoder: Array
    supported_rank: int
    retained_rank: int
    certificate_type: str
    certificate_value: float | None
    diagnostics: dict
```

This is an API specification, not executable MACE code: `Array`, imports, serialization, device handling, and checkpoint adapters must be implemented. Each array must additionally carry axis metadata in storage. Validate shapes and reject irrep or interface mismatches rather than silently reshaping tensors.

### 17.3 Primary discovery pseudocode

```text
INPUT:
  trusted, version-pinned checkpoint
  supported architecture adapter
  fixed structural measures
  candidate rank allocations

1. Inspect the checkpoint without changing its parameters.
2. Write the full interface and operation registry.
3. Verify the original readout split and supported polynomial operations.
4. Construct and verify the exact symmetrized path quotient.
5. Extract the relevant radial functions at the precisely identified interface.
6. Verify their multineighbor reconstruction and derivatives.
7. For each central species and irrep block:
     a. Integrate the declared radial overlaps with convergence checks.
     b. Find the supported function space and canonical coordinates.
8. Compose the entire first-readout branch into symmetric K_1, K_2, K_3.
9. Verify the complete coefficient evaluator against the original branch.
10. Contract full cross-path marginals to form Γ.
11. Check symmetry, positive semidefiniteness, and equivariant block structure.
12. Diagonalize multiplicity blocks and allocate ranks under the declared budget.
13. Build native encoders/decoders and projected coefficient tensors.
14. Compute the proved marginal-tail bound and actual coefficient residual.
15. Export mode cards, body-order kernels, and the branch evaluator.
16. Freeze outputs before examining benchmark-reference outcomes.

OUTPUT:
  decomposition artifacts
  source/checkpoint/measure manifests
  exactness and numerical diagnostics
  no claimed chemical mechanism until evaluation is complete
```

### 17.4 Reference bridge pseudocode

```text
assert T's external partition has a true separator H
L = U @ U.T
R = V.T @ V
C, support = supported_square_root(L)
S = C.T @ R @ C
lambda, Z = symmetric_eigendecomposition(S, descending=True)
B = C @ Z[:, :rank]
D = Z[:, :rank].T @ supported_left_inverse(C)
P = B @ D
T_reduced = V @ P @ U
check ||T - T_reduced||_F^2 == sum(lambda[rank:])
```

For large implicit $U$ and $V$, the purpose of tensor environments is to compute $L$ and $R$ without materializing them. The assertion about a true separator is a mathematical obligation, not a software flag that makes it true.

### 17.5 First-branch marginal pseudocode

```text
Gamma = zero_operator_on_canonical_primitive_space()
for nu in (1, 2, 3):
    K = composed_symmetric_kernel[nu]
    for slot in range(nu):
        Gamma += beta[nu] * contract_all_other_slots(K, K, slot)

blocks = verify_and_extract_irrep_multiplicity_blocks(Gamma)
Pi = choose_equivariant_projector(blocks, rank_budget)
for nu in (1, 2, 3):
    K_projected[nu] = apply_on_every_slot(K[nu], Pi)

actual_error2 = sum(beta[nu] * norm2(K[nu] - K_projected[nu]))
bound = trace((Identity - Pi) @ Gamma)
assert actual_error2 <= bound + numerical_tolerance
```

The factorized backend and the dense oracle must implement the same expression. An implementation that sums individual path norms instead of contracting cross-path terms is a different method and should fail cancellation tests.

### 17.6 Configuration template

```yaml
project: mace-odt
status: proposed
checkpoint:
  path: null                 # Required: a trusted original file.
  sha256: null               # Required: computed from that exact file.
  source_revision: null      # Required: implementation used to load/evaluate it.
  adapter: null              # Required: audited and tested architecture adapter.

discovery:
  target: first_linear_energy_branch
  central_species_mode: separate
  construction_data: none
  structural_seed: 260921
  floating_point: float64
  quotient_exact_paths: true
  use_molecular_activation_covariance: false
  polynomial_head_shadow: false

measure:
  radial_interval_angstrom: null   # Required: fixed before inspecting spectra.
  radial_coordinate: r_over_rc
  radial_weight: normalized_uniform
  species_weighting: uniform_over_declared_neighbor_species
  angular_measure: normalized_haar
  harmonics: converted_to_orthonormal_under_declared_measure
  order_weights: {1: 1.0, 2: 1.0, 3: 1.0}
  quadrature_order: 128            # Initial proposal; convergence is mandatory.
  quadrature_refinements: [256, 512]

method:
  primary: equivariant_tied_mixed_order_marginals
  reference: dense_tiny_coefficients
  optional_refinement: none
  rank_candidates: [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96]
  rank_allocation: per_irrep_and_declared_cost_budget
  max_intermediate_bytes: 2000000000  # Example safety cap, not a required device.

validation:
  gauge_condition_numbers: [1, 3, 10, 30, 100]
  gauge_repeats: 10
  synthetic_identity_relative_tolerance: 1.0e-10
  real_checkpoint_tolerances: calibrate_against_float64_baseline
  reference_labels_used_for_discovery: false
  export_scope: branch_only

benchmark:
  dataset_manifest: null      # Required before an empirical study.
  split_manifest: null        # Required and frozen before selection.
  chemical_case_manifest: null
  reference_error_is_distinct_from_fidelity: true
```

All numerical defaults here are engineering proposals, not reported experimental settings or established optimal choices. Invalid rank candidates are removed according to actual supported multiplicities. The radial interval, checkpoint identifiers, and benchmark details are intentionally unresolved rather than invented.

### 17.7 Provenance and deterministic execution

Store the checkpoint hash, package versions, exact source revision, hardware, numerical precision, integration settings, random seeds, and configuration hash with every result. Current public implementation code can guide inspection but cannot substitute for the exact version that produced a released checkpoint.

Use a trusted checkpoint source and an explicit load procedure; do not automatically execute arbitrary code from unknown model files. Preserve the original checkpoint unchanged and write outputs to new versioned files.

Repeated runs should reproduce spectra and invariant subspaces within calibrated numerical tolerance. Failure near a degenerate eigenspace should be reported as subspace ambiguity, not hidden with an arbitrary matching rule.

---

<a id="section-18"></a>

## 18. Validation gates and test suite

### 18.1 Gate structure

A result advances only after passing the preceding exactness gates. Attractive spectra never override a failed reconstruction or transport test.

| Gate | Required evidence | Stop or downgrade condition |
|---|---|---|
| G0: architecture | Exact module graph, activation inventory, target and interface registry. | Unknown feature-dependent operation, ambiguous interface, or unsupported checkpoint. |
| G1: quotient | Coefficient-level nullspace identity and positive controls. | Reported source dimensions assumed without reconstruction. |
| G2: radial functions | Multineighbor reconstruction, angular convention, derivative agreement. | Two-atom agreement only, or wrong pre/post-map interface. |
| G3: branch compiler | Original and coefficient branch agree in energy and derivatives. | Residual error above calibrated numerical/representation floor. |
| G4: canonical metric | Supported-space reconstruction and quadrature convergence. | Rank changes erratically with integration or precision. |
| G5: decomposition | Dense oracle and factorized marginals agree; proved tail inequality holds. | Cross-path cancellation lost or eigensolve inconsistent. |
| G6: gauge | Invariant function images and appropriately transported edits. | Only raw eigenvalues or arbitrary coordinate vectors compared. |
| G7: export | Serialized evaluator reproduces the declared projected target. | Shadow/native/branch scopes conflated or reload differs. |
| G8: interpretation | Matched controls and held-out behavioral evidence. | A descriptive picture is the only evidence of a mechanism. |

### 18.2 Original χ-net reference tests

Use very small dimensions and depths so that the complete independent-slot coefficient tensor can be constructed exactly. The test should include nontrivial symmetric cores, an appended constant coordinate, tied layer copies, and a nontrivial output map.

For each layer, verify:

- RQ preserves the unfolded tensor and the folded polynomial;
- the lower subtree has the claimed row-isometric matricization;
- the top-down message includes the sibling partial trace;
- its eigenvalues equal squared singular values of the explicit cut matricization;
- truncation of one occurrence has the correct discarded-eigenvalue residual;
- refolded shared truncation matches the corresponding simultaneous insertion;
- any claimed global bound uses the correct occurrence count and singular-value convention;
- input-side truncation is not assumed to preserve a previously established isometry.

An optional SVHN reproduction can test the published empirical pattern, but is separate from these algebraic gates. It requires the original training and effective-dimension conventions; the paper alone does not make every implementation detail unambiguous. [D:AA–AG]

### 18.3 MACE architecture and radial tests

Reproduce the HTML's two-readout identity with the exact checkpoint scaling and reference energies. Extract the radial curves, then reconstruct the entire designated interaction interface on multineighbor environments, not just the two-atom extraction inputs.

Check changes of orientation, neighbor ordering, species, radial position, and neighbor count. Compare radial derivatives and resulting coordinate gradients. Use proper rotational and parity transformations where the model includes them.

The HTML reports a radial reconstruction gate of $10^{-10}$ and residuals near $10^{-15}$ for its setup. Treat these as source measurements, not a universal tolerance on every device and package version. Establish the local floating-point baseline and record both absolute and relative errors. [H:appG-curves; H:appI]

### 18.4 Quotient tests

Test both positive and negative cases: retained path perturbations should usually change the appropriate tensor, while mathematically null perturbations should not. For a positive control, explicitly select a perturbation whose composed tensor is nonzero rather than relying on chance.

Add large null perturbations without changing the model function, then verify invariance of the composed $K_\nu$, its canonical spectrum, and its functional mode images. This is stronger and better typed than expecting a channel-space spectrum to contain exactly fifteen zero modes.

### 18.5 Gauge tests

Separate two classes of transformation:

**Native legal gauges:** transformations verified to leave the checkpoint's actual module class and predictions unchanged.

**Analysis-graph reparameterizations:** arbitrary invertible maps inserted with exact inverses in an enlarged tensor representation. These may be valid changes of representation without being expressible as the original channel-diagonal MACE parameterization.

Both are useful tests, but only the first supports a statement about the HTML's native gauge family. For each, compare full functions, supported projectors, encoded features, radial images, spectra, and export behavior. Include uncompensated transforms as controls.

### 18.6 Numerical support tests

Construct exact duplicate radial functions and nearly dependent functions in toy models. Verify exact supported-space reduction, the distinction between quotienting and approximation, and the reported uncertainty of near-null modes.

Apply non-orthogonal gauges of increasing condition number. A numerically fragile decomposition may remain algebraically correct but require higher precision or a smaller supported space. Report this limitation rather than treating an arbitrary $\epsilon I$ regularizer as invariant.

### 18.7 Body-order and force tests

For random small symmetric polynomial branches, verify (32) against (33). Confirm vanishing of distinct-neighbor terms above order three. Test the branch evaluator and projected evaluator with the same basis and normalization.

For a differentiable radial implementation, compare analytic or automatic gradients with a small number of carefully scaled finite-difference checks. Finite differences are a debugging tool, not the production force definition. Check the sign convention, units, atom ordering, and total-energy summation.

### 18.8 What has actually been executed for this specification

The accompanying `verify_mace_odt_math.py` implements deterministic finite-dimensional checks with NumPy. The accompanying JSON report records their outcomes. These tests verify identities and construct counterexamples to overbroad claims; they do not implement MACE adapters, run ODT on a real checkpoint, reproduce SVHN, or establish empirical compressibility.

The final appendix lists the exact executed checks and selected numerical results. All MACE-specific gates in this section remain implementation work.

---

<a id="section-19"></a>

## 19. Empirical program

### 19.1 E0 — Establish the original ODT baseline

**Question:** does the reference implementation reproduce the actual valid-cut decomposition in a binary χ-net?

**Design:** first run the dense small-tree tests. If empirical reproduction is pursued, match the published SVHN setting as closely as available artifacts permit. Record deviations in training, noise, normalization, width, and effective-dimension calculation.

**Outcome:** a verified baseline and a clear statement of which reported source observations were reproduced. It is not necessary to achieve a particular MACE result to pass this experiment.

### 19.2 E1 — Integrate the HTML's exact redundancies

**Question:** do the supported checkpoint's coupling-path null spaces agree with its actual symmetrized tensor map?

**Design:** reconstruct the path map; verify source-compatible ranks; convert a copy of the checkpoint; perturb exact null and retained coordinates; compare original and converted outputs and the new decomposition.

**Primary output:** an audit table by layer, species, correlation order, and output irrep. Include stored dimension, supported path dimension, exactness residual, and whether the checkpoint already uses a reduced CG representation.

**Interpretation:** this establishes the algebraic starting point. It is not the main discovery of learned low rank. [H:nullspace; H:appC-conversion]

### 19.3 E2 — Reconstruct the radial interface and first branch

**Question:** does the new functional representation compute the original first branch?

**Design:** extract curves and compile $K_1,K_2,K_3$. Validate on deterministic synthetic environments and an evaluation-only set of geometries. Include both ordinary and deliberately difficult configurations within the declared domain.

**Primary output:** energy and derivative residuals before any truncation, quadrature convergence, and exact body-order identities. A failure here invalidates later compression curves.

### 19.4 E3 — Compare local and composed spectra at nontrivial interfaces

**Question:** does downstream composition change the ranking and apparent effective dimension beyond an architectural endpoint ceiling?

**Design:** compare local weight spectra, radial-curve spectra in native coordinates, canonical global marginals, and dense global cut spectra where an oracle is feasible. Report by central species and irrep, with the target norm stated for each.

**Required controls:** remove or separately report constant/reference contributions; compare with architecture-matched random or untrained controls when available; report exact support ceilings; repeat under allowed gauge changes.

**Primary output:** spectra plus mode-overlap and tail curves. This is descriptive evidence, not a compression result by itself.

### 19.5 E4 — Rank versus first-branch fidelity

**Question:** at a matched rank budget, do composed-weight modes preserve the branch better than local alternatives?

**Design:** evaluate a frozen rank ladder such as the candidates in Section 17, intersected with each supported multiplicity. Use the same export path and the same evaluation configurations for all methods. Preserve complete irrep multiplets.

Measure the actual coefficient error, its upper bound, energy error, force error, and conformational energy-difference error. Include both single-interface and explicitly specified multi-interface projections; do not infer the latter from the former.

**Decisive result:** a reproducible improvement in the rank required to meet a preregistered fidelity tolerance, not just a lower-looking effective-rank statistic.

### 19.6 E5 — Branch-only preservation versus shared-model intervention

**Question:** does a subspace sufficient for the first energy branch also preserve computations that share its features?

Use two distinct evaluators:

$$
 E^{\mathrm{branch}}_{\mathrm{comp}}(X)
 =E_0(X)+E_{L,\mathrm{comp}}(X)+E_{N,\mathrm{original}}(X),
\tag{45}
$$

and a **shared-model** evaluator that inserts the projector at the actual shared interface and lets every consumer receive the altered features.

Equation (45) isolates first-branch approximation. It may require keeping the original nonlinear branch's computation and therefore need not reduce inference cost. The shared-model edit is a stronger intervention with no automatic first-branch-only certificate.

**Primary output:** both fidelity curves and their difference. A large difference is evidence that the nonlinear branch uses information the linear branch does not; it is not a reason to hide the branch-only scope.

### 19.7 E6 — Gauge and metric robustness

**Question:** which reported objects are invariant, and which depend on structural choices?

**Design:** rerun discovery after compensated native gauges and analysis-only reparameterizations. Separately vary a small preregistered set of radial measures, radial intervals, species weights, and order weights.

**Expected distinction:** exact gauge covariance is an algebraic requirement. Stability across genuinely different measures is an empirical robustness property and need not hold exactly.

**Primary output:** functional-subspace agreement, eigenvalue changes under gauges, and rank/fidelity sensitivity under metric changes. Do not merge these into one “invariance” statistic.

### 19.8 E7 — Revisit the HTML's curve-mode/benchmark-effect separation

**Question:** do composed-weight subspaces explain anything beyond the local curve content of the benchmark-selected directions?

**Design:** reproduce the HTML's curve-SVD ladder and specified intervention rules when the original direction tensors, model, and benchmark artifacts are available. Include its rungs 3, 5, 20, and 45 for replication, then use a denser predefined ladder for the new study.

Compare the selected directions' functional images, their projections onto global versus local subspaces, and their benchmark effects under fully specified encoders/decoders. Include random directions in the same canonical subspace, random directions matched on leading-mode mass, and full-space random controls with matched intervention scale.

**Crucial distinction:** the HTML's positive effect is an improvement in error against reference energies after removal. ODT's principal objective is preservation of the original model function. A direction can be important for reproducing the model and harmful to its reference accuracy. Neither outcome contradicts the other objective. [H:selection; H:deflation; H:calibrations]

**Primary output:** a joint analysis of local representation, global coefficient importance, original-model fidelity, and reference-error change. This experiment must not be summarized as “ODT fixes the HTML's SVD failure” unless that specific claim is supported.

### 19.9 E8 — Mechanistic case studies with the HTML's controls

**Question:** can a small mode interaction or subspace account for a specified model-specific chemical effect?

**Design:** choose cases before inspecting all mode pictures. For the first branch, use effects the readout split places there. Render radial images, reduced-core interactions, and body-order difference kernels. Test targeted and matched-control interventions.

A successful explanation must predict a previously untested change of geometry or species within its stated scope, not merely describe an already inspected curve. Include an alternative explanation based on edit support, amplitude, or the stored linear-readout coefficient and test whether it suffices.

**Primary output:** a compact, falsifiable account of one effect with control results and limits. Failure to obtain one is compatible with a valid compression paper.

### 19.10 E9 — Energy differences, forces, and difficult geometries

**Question:** does low coefficient error preserve the quantities for which a potential is used?

**Design:** evaluate conformer differences, torsional scans, coordinate perturbations, and relevant strain or stress quantities for the chosen checkpoint domain. Include held-out molecular groups and challenging short-range configurations only within clearly stated validity intervals.

Report absolute errors and scale them against meaningful variations of the original model, not enormous atomic reference-energy offsets. A small relative error in total energy dominated by reference constants is not convincing evidence of force or conformational fidelity.

**Primary output:** paired energy/force plots, quantiles and maxima, and explicit failure regions. Optional short dynamical rollouts are validation-only and require a separate stability protocol.

### 19.11 E10 — Full polynomial observables and exact nonlinear head

**Question:** can the second-layer polynomial target be represented and compressed efficiently enough to preserve the original nonlinear head?

**Design:** complete the two-layer gates, then compare preactivation-preserving methods using the bounds in Section 16 with full-energy and force measurements. Include endpoint-rank baselines and a plain local head-map factorization.

**Primary output:** contraction costs, approximation errors, preactivation-domain checks, and full-model fidelity. This is a separate experimental tier, not a required success condition for an exact first-branch paper.

### 19.12 E11 — Polynomial-head shadow study

**Question:** does a certified shadow give a tractable and stable decomposition of the full energy target?

**Design:** sweep a predetermined set of approximation degrees and domains. Report approximation errors before any spectral truncation, then separate projection and export errors. Compare shadow and projected-original evaluators.

**Primary output:** a degree/domain/rank/cost tradeoff and mode-subspace stability analysis. If the shadow's coefficient norm grows or changes substantially while energy accuracy improves, report that explicitly.

### 19.13 E12 — Actual compression, storage, and runtime

**Question:** does the exported representation save anything in practice?

**Design:** benchmark original, quotient-only, branch-only, projected-native, and compiled-compressed evaluators separately. Use realistic atom counts and neighbor densities, identical precision, warmup, synchronization, and the same requested outputs.

Measure energy-only inference, energy-plus-forces evaluation, peak memory, model-file size, number of stored learned values, number and size of fixed CG tensors, and total arithmetic. Include the one-time decomposition cost separately.

A dense transformed core can erase the benefit of a smaller bond. A low-rank branch evaluator running alongside the original full backbone may be slower. Both are possible outcomes. The HTML's quotient-only timing gains are an existing comparator, not evidence for new ODT speedups. [H:appC-conversion; H:appI]

### 19.14 E13 — Architecture versus learned structure

**Question:** is any additional compactness specific to trained composition?

**Design:** compare released or independently trained models with compatible untrained/random controls, accounting for scale, output normalization, constant contributions, and architectural rank ceilings. A shuffled-weight control may destroy many things at once, so interpret it narrowly.

When feasible, compare multiple training seeds and model sizes using common functional-space alignment. Do not infer the origin of rank differences from one trained checkpoint and one poorly scaled random draw.

**Primary output:** evidence distinguishing architecture-forced support, training-associated spectral concentration, and metric-induced apparent compactness.

---

<a id="section-20"></a>

## 20. Baselines and matched controls

### 20.1 Baseline set

| Baseline | Purpose | Required caution |
|---|---|---|
| Native local weight SVD | Establish whether individual parameters already expose the structure. | Coordinate dependent; state matricization and update every affected consumer consistently. |
| Native radial-curve SVD | Reproduce the HTML's local functional-norm ranking. | Its ordinary Euclidean channel SVD is not automatically invariant to full non-orthogonal channel changes. |
| Canonical random subspace | Test whether the choice within a fixed function space matters. | Draw with respect to the declared metric; retain full irrep multiplets. |
| Dense exact global cut SVD | Verify the implicit decomposition on tiny tractable instances. | An oracle, not a scalable competing method. |
| Readout-omitted or locally weighted marginal | Test whether downstream composition adds value. | Define its metric and normalize comparisons; this is an ablation. |
| Spectral initialization plus coefficient optimization | Test how much the upper-bound method loses to optimizing the tied objective. | Additional compute and optimization are part of the method; no claim of an exact spectral optimum. |
| Data PCA or activation-aware compression | Compare against a strong distribution-specific alternative. | Mark the additional data access and fitting budget. |
| SAE-based or benchmark-selected directions | Connect to the HTML's interpretation study. | Match intervention rules, reconstruction quality, data access, and selection protocol. |
| Exact path quotient only | Separate existing architectural redundancy from new approximation. | No approximation error should be charged to a correct exact quotient. |
| Terminal readout/head factorization | Control for obvious output-width ceilings. | Not a deep mechanism discovery. |

After complete upstream whitening, upstream covariance is the identity. There is no unique “leading whitened upstream PCA” direction to compare with: all supported directions are tied under that local norm. Do not use numerical noise in this identity as a meaningful baseline ranking.

### 20.2 Fairness has several dimensions

Compare methods at matched retained multiplicities and at matched measured resource cost. These comparisons answer different questions and should both be reported when compression is claimed.

Hold the original checkpoint, target observable, evaluation split, numerical precision, export interface, and intervention support fixed. A comparison between one method's first branch and another method's complete potential is not meaningful without a clear decomposition of the targets.

### 20.3 HTML-inspired control families

**Representation control.** Compare radial-image agreement with directions matched on their mass in leading curve modes. High image correlation can arise from a shared dominant component.

**Detection control.** If semantic labels are used, compare against direct probes and label-table/random-embedding controls of the appropriate kind. Label recoverability alone does not show a feature is used causally.

**Support control.** Hold the edited atoms, layers, and channels' reach fixed. The source's reach-only AUC identity explains why support can produce selectivity independently of direction semantics.

**Linear-readout control.** Compute the stored coefficient $s w^\top b_k$ for a constant activation edit. An observed linear-branch steering slope matching this coefficient is an algebraic sanity check, not novel mechanistic evidence.

**Dual control.** Compare canonical orthogonal removal, native Euclidean orthogonal removal, and any covariance-derived oblique removal as distinct edits. Match their actual effect scale as well as their named direction.

**Random-subspace control.** Draw random directions or subspaces inside the same canonical support and, where relevant, inside the same leading-mode span.

**Cross-seed control.** Separate algorithmic gauge ambiguity, unstable spectral eigenspaces, different learned functions, and SAE seed instability. These are not interchangeable explanations. [H:detection; H:selectivity; H:steering; H:deflation; H:calibrations]

---

<a id="section-21"></a>

## 21. Metrics and statistical reporting

### 21.1 Rank is not one number

For each reported spectrum specify whether it contains singular values $\sigma_j$, squared singular values $\lambda_j$, or aggregate scores from $\Gamma$.

Suggested standardized descriptive statistics for a positive-semidefinite operator $G$ with eigenvalues $\lambda_j$ are

$$
 r_{\mathrm{stable}}=
 \frac{\sum_j\lambda_j}{\max_j\lambda_j},
 \qquad
 r_{\mathrm{participation}}=
 \frac{(\sum_j\lambda_j)^2}{\sum_j\lambda_j^2},
\tag{46}
$$

and entropy effective dimension

$$
 r_{\mathrm{entropy}}=
 \exp\!\left(-\sum_j p_j\log p_j\right),
 \qquad p_j=\lambda_j/\sum_k\lambda_k.
\tag{47}
$$

For a zero operator, mark these statistics as undefined or give an explicitly stated zero-rank convention. For a true Gram $G=T^\top T$, the first is the usual stable rank of $T$. For $\Gamma$, it is only the corresponding statistic of the aggregate score spectrum.

Define a spectral mass rank by

$$
 r_{\epsilon}=\min\left\{r:
 \sum_{j>r}\lambda_j\le\epsilon^2\sum_j\lambda_j\right\}.
\tag{48}
$$

For a single true cut, this corresponds to a relative coefficient approximation criterion. For the mixed-order aggregate, distinguish normalization by $\operatorname{tr}\Gamma$ from normalization by $\sum_\nu\beta_\nu\|K_\nu\|^2$: the former counts slot occurrences and is not the same relative error criterion.

Finally report the **measured fidelity rank** needed to meet energy, force, or relative-energy tolerances. Do not identify this with any effective-dimension statistic or with Dooms' reported L2 effective dimension without matching its definition.

### 21.2 Energy and force fidelity

For a configuration set $\mathcal X$, use

$$
 \mathrm{RMSE}_{E,\mathrm{fid}}
 =\left[\frac1{|\mathcal X|}\sum_X
 (E_{\mathrm{comp}}(X)-E_{\mathrm{orig}}(X))^2\right]^{1/2},
\tag{49}
$$

with a separately reported per-atom normalization when appropriate. For forces,

$$
 \mathrm{RMSE}_{F,\mathrm{fid}}
 =\left[
 \frac{\sum_X\sum_{i=1}^{N_X}
 \|F_{\mathrm{comp},i}(X)-F_{\mathrm{orig},i}(X)\|^2}
 {3\sum_XN_X}
 \right]^{1/2}.
\tag{50}
$$

Also report molecule-balanced force error so that large molecules do not silently dominate every statistic. Use quantiles and maxima, not only a mean.

For a conformer pair $X,Y$, report

$$
 \Delta_{\mathrm{rel}}(X,Y)=
 [E_{\mathrm{comp}}(X)-E_{\mathrm{comp}}(Y)]
 -[E_{\mathrm{orig}}(X)-E_{\mathrm{orig}}(Y)].
\tag{51}
$$

This directly tests whether the model's energy difference is preserved. Report reference-label accuracy separately, using $E_{\mathrm{ref}}$ rather than $E_{\mathrm{orig}}$.

### 21.3 Semantic concentration and interaction metrics

Possible descriptive measures include radial localization, element-pair norm concentration, angular-order concentration, reduced-core sparsity, and body-order cancellation. Define thresholds and measures before looking for appealing examples.

Concentration can be created by the basis, species weighting, normalization, or the architecture itself. Compare to matched random functions in the same support. The HTML reports failures of several localization and random-direction hypotheses; a semantic concentration statistic should therefore not be interpreted without its null distribution. [H:appH]

### 21.4 Splits and selection

Separate four stages: algebraic implementation tests, weight-only discovery, model-fidelity validation, and final held-out evaluation.

Metrics and rank candidates should be fixed before examining final reference-label outcomes. If rank is selected using validation molecules, describe the final rank as validation-selected even when the candidate subspaces were discovered from weights alone.

Use splits by molecular formula, scaffold, or another scientifically justified group. For reaction paths, keep all frames of a path together and avoid leakage between related molecules. The HTML's benchmark selection sequence is a source replication target, not permission to select directions on the final test set of a new study.

### 21.5 Uncertainty and independent units

Bootstrap or otherwise quantify uncertainty at the level of molecules, paths, or independent groups rather than treating every correlated atom or trajectory frame as independent. Compare methods pairwise on the same groups.

Report the number of independently trained checkpoints and the number of random-subspace seeds separately from the number of evaluated geometries. A large geometry count from one model does not provide uncertainty over training runs.

A rank advantage should include uncertainty in the achieved fidelity and a predefined rule for crossing the tolerance. Interpolating a threshold between sparsely sampled ranks requires qualification. In particular, the HTML's recovery at rank 45 with the preceding rung at 20 is not a measurement of a sharp intrinsic threshold at 45. [H:limitations]

### 21.6 Predeclared success criteria

Before large experiments, specify numerical fidelity tolerances in physical units, eligible interfaces, comparison baselines, and the minimum practically meaningful improvement. Choose tolerances from the intended modeling task and original-model error scale, not from the rank curve after seeing it.

A useful example criterion is: **under a fixed full-irrep rank budget, the composed method achieves a lower held-out force-fidelity error than local alternatives, with a paired uncertainty interval that excludes the predefined negligible-difference region**. The exact region and sample sizes remain to be set for the selected dataset.

No mandatory criterion should require discovering ten modes, outperforming every data-aware baseline, or producing a named chemical mechanism. Different levels of evidence support different papers.

---

<a id="section-22"></a>

## 22. Adversarial review and rejection criteria

The following table is part of the research design. A failure must change the claim, implementation, or scope; it must not be explained away by emphasizing an attractive visualization.

| Potential failure | Adversarial test | Required response |
|---|---|---|
| The method is merely local radial SVD with new terminology. | Remove or vary downstream weights while holding radial functions fixed; compare resulting subspaces. | Demonstrate downstream dependence through complete coefficients, or label the method local. |
| A “global bond” is not a separator. | Explicitly draw the external partition and all remaining connecting indices. | Use the actual separator or general environment; do not invoke the two-sided theorem. |
| The source Gram recurrence was mistranslated. | Compare shapes and results with dense cut matricizations. | Include the sibling partial trace and rederive the contractions. |
| Eigenvalues are treated as singular values. | Check residuals against $\sum\lambda_j$, $\sum\lambda_j^2$, and explicit SVD. | Use the correct convention and rewrite the certificate. |
| Input projection invalidates canonical form. | Recompute $QQ^\top$ after truncation. | Re-orthogonalize before using isometric cancellations. |
| The formal lift includes a polynomial that vanishes on repeated inputs. | Add the zero-polynomial construction in Section 6 or another exact rewrite. | Restrict identifiability to the fixed lift or construct an appropriate symmetric/functional quotient. |
| Physical repeated inputs are treated as independent. | Compare exact higher moments with products of second moments. | Declare a coefficient norm or construct the correct correlated metric. |
| Channel spectra are changed by harmless scaling. | Apply compensated non-orthogonal gauges. | Correct whitening and environment transport; do not claim raw-Gram invariance. |
| Singular whitening claims covariance outside its support. | Test both reachable and arbitrary native vectors. | State the supported-space result and any off-support convention. |
| Fifteen path-null directions are mistaken for fifteen channel-null directions. | Inspect the domain and codomain of each linear map. | Keep the path quotient and channel decomposition distinct. |
| New low rank is an output bottleneck. | Compare against scalar-readout and head-width ceilings. | Move the result to a nontrivial earlier interface or classify it as architectural. |
| A coefficient spectrum is dominated by constants. | Remove or separately report reference/constant terms and recompute metrics. | Report interaction-only and full-target results separately. |
| Pathwise norm accumulation loses cancellation. | Add equal-and-opposite path tensors. | Contract all cross terms before scoring. |
| Separate central-species spaces are mislabeled one shared model rank. | Count all stored maps and evaluate shared versus conditioned projectors. | Report conditioned ranks and resource cost honestly. |
| A common projector is assumed to be the exact optimum. | Compare the actual tied objective with the marginal bound and optimized competitors. | Claim upper-bound optimality/quasioptimality only, or report the optimizer separately. |
| Rotation or parity symmetry is broken. | Transform inputs and compare outputs, forces, and intermediate irreps. | Project only admissible multiplicity spaces and retain whole multiplets. |
| The HTML's curve covector is confused with an intervention vector. | Check transformation laws and the encoder–decoder pairing. | Export a covector and its specified dual decoder; correct the weight edit. |
| A mode looks chemical but has no specific functional role. | Compare with random functions in the same support and matched interventions. | Keep a descriptive representation claim, not a mechanism claim. |
| Benchmark improvement is confused with model preservation. | Report both original-model fidelity and reference-label error changes. | State the two objectives separately; no inferred causal link from rank alone. |
| A first-branch projection is presented as full-model compression. | Compare branch-only and all-consumer shared edits. | Restrict scope or analyze the nonlinear branch as well. |
| An apparently tiny energy error hides a force failure. | Evaluate gradients, conformational differences, and short-range cases. | Add derivative-aware requirements or withdraw practical-fidelity claims. |
| A head shadow is accurate only on the original activation range. | Bound and inspect compressed preactivations too. | Enlarge the certified domain, reduce edits, or withdraw the certificate. |
| Shadow function convergence is confused with mode convergence. | Compare degrees under a common functional metric and inspect spectral gaps. | Report stable function/subspace results, not unsupported individual-mode limits. |
| Tensorization is mathematically exact but computationally unusable. | Record contraction width, peak memory, and actual factorized costs. | Narrow the target or use an explicitly approximate method with an error budget. |
| Smaller bonds create denser, slower cores. | Benchmark exported energy and force evaluators, not only parameter counts. | Separate effective-rank findings from engineering compression claims. |
| A purported chemical mechanism is a fragment extrapolation artifact. | Compare fragment kernels with full-molecule effects and cancellation diagnostics. | State the model-specific identity and avoid physical-force attribution. |
| Selection leaked into the held-out benchmark. | Audit direction, rank, metric, and case-selection timestamps against data access. | Rerun on a new independent split or label the analysis exploratory. |
| A negative result is overgeneralized. | Compare scope, architecture, checkpoint, metric, and reference implementation. | Limit conclusions to the tested construction; do not declare all weight interpretation ineffective. |
| Novelty is claimed for standard HSVD/whitening. | Compare theorems and algorithms with primary tensor-decomposition literature. | Attribute the mathematics and locate novelty in the specialization or scientific finding. |

### 22.1 Stronger falsification experiments

Three tests are especially valuable because they attack the interpretation of the result rather than its numerical implementation.

**Function-preserving rewrite test.** Produce two representations that agree on the intended physical function but differ as formal lifts. Determine which method outputs agree. This establishes the actual equivalence class under which the method is identifiable.

**Same-spectrum, different-behavior test.** Construct toy kernels with identical aggregate marginal spectra but different coupled coefficients. Their pointwise or intervention behavior can differ. This prevents interpreting the spectrum alone as a complete mechanism description.

**Fidelity-versus-error-correction test.** In a controlled synthetic setting with a known reference function, add a model error lying in a spectrally important direction. A faithful decomposition may preserve that error. This demonstrates why discovering model mechanisms and improving a benchmark are separate objectives.

These are proposed experiments beyond the executed companion checks. They should be added to a full research implementation rather than described as completed here.

### 22.2 What would invalidate the primary method

The first-branch construction should be considered invalid as implemented if it cannot reproduce the original branch before truncation, fails its own coefficient bound beyond numerical tolerance, or fails covariance under the transformations it claims to support.

If it passes those tests but offers no useful rank advantage, it is a valid method with a negative empirical outcome. If it offers rank advantage but not chemical specificity, it supports functional compression or effective-rank analysis, not mechanistic attribution.

---

<a id="section-23"></a>

## 23. Milestones, dependencies, and scope decisions

### 23.1 Milestone M0 — Source and checkpoint audit

**Deliverables:** immutable source registry, checkpoint manifest, module inventory, interface registry, and a table mapping every operation to exact polynomial, exact primitive function, or unsupported/nonlinear status.

**Exit condition:** a precise first-branch target and at least one nontrivial admissible interface have been identified. Unknown checkpoint details are resolved from actual artifacts, not inferred from generic architecture descriptions.

### 23.2 Milestone M1 — Algebraic reference implementation

**Deliverables:** original χ-net reference, dense bridge oracle, mixed-order projection implementation, and all mathematical counterexample tests.

**Exit condition:** exact cut spectra, transformation laws, and tail inequalities agree with explicit tensors on small cases. The proof scope and the implementation scope coincide.

### 23.3 Milestone M2 — HTML reconstruction and first-branch compiler

**Deliverables:** exact path quotient, radial-function evaluator, controlled overlap integration, first-branch kernels, and two independent body-order evaluators.

**Exit condition:** the original branch's energy and forces are reconstructed within a documented error budget. The HTML's source-specific counts and floors are reproduced only where the setup matches.

### 23.4 Milestone M3 — First-branch scientific evaluation

**Deliverables:** matched local/global/random baselines, spectra, rank–fidelity curves, gauge tests, metric sensitivity, and architecture controls.

**Exit condition:** the empirical outcome is known well enough to choose an honest paper direction. There is no requirement that low rank appear.

### 23.5 Milestone M4 — Controlled semantic analysis

**Deliverables:** mode/subspace cards, reduced interaction cores, body-order difference kernels, and preregistered chemical cases with HTML-inspired controls.

**Exit condition:** either a specific falsifiable interpretation survives the controls, or the result is explicitly limited to representation/compression.

### 23.6 Milestone M5 — Full-model extension

**Dependencies:** M0–M3, an efficient two-layer polynomial representation, and an explicit choice between the exact-head and shadow-head routes.

**Deliverables:** full observable definition, contraction strategy, supported error statement, and full-model evaluation. This may form a separate project if its complexity dominates the first-branch work.

### 23.7 Milestone M6 — Engineering export

**Deliverables:** reloadable compressed evaluator, measured runtime and memory, and a comparison with quotient-only conversion.

**Exit condition:** actual resource benefits are measured. A mathematically reduced representation that is slower still remains a valid analysis artifact, but not a speedup result.

### 23.8 Required resources not supplied by the present documents

The provided materials do not contain a complete executable MACE checkpoint adapter, all original SAE direction tensors, the exact benchmark split files, or all source experiment scripts. An implementation must obtain or reconstruct them and record the provenance.

The Markdown concept note is a design input, not an empirical source. The HTML is a manuscript with reported measurements, not a substitute for its raw experimental artifacts. The current public MACE source is useful for architectural inspection but is not evidence that a historical checkpoint used every current option.

### 23.9 Decision matrix after first-branch evaluation

| Observed outcome | Defensible conclusion | Next step |
|---|---|---|
| Global selection substantially reduces the fidelity rank and yields controlled semantic structure. | Composed weights expose useful, physically inspectable subspaces beyond local selection in the tested branch. | Extend to shared/full-model targets and compile an efficient evaluator. |
| Global selection improves fidelity rank, but modes are semantically mixed. | A functional compression/effective-rank result, without an individual-mechanism claim. | Study joint cores and report the limit honestly. |
| The spectrum is concentrated, but force fidelity remains poor. | The chosen coefficient metric misses behaviorally important directions or is too weak for derivatives. | Examine bounds, derivative-aware targets, and metric sensitivity. |
| No improvement beyond local SVD after exact quotienting. | No evidence for the hypothesized additional benefit in this tested target and norm. | Analyze architectural ceilings, other targets, and whether the negative result is informative. |
| First branch compresses but shared-model fidelity fails. | The two readouts use different information at the shared interface. | Target the nonlinear branch's polynomial observables. |
| Full tensor contraction is intractable at the intended scale. | The representation exists algebraically, but the proposed exact backend is not computationally viable there. | Narrow the target or introduce a separately controlled approximate contraction. |
| Gauge tests fail. | The implementation or its stated identifiability claim is wrong. | Fix the mathematics or interface mapping before reporting spectra as invariant. |

---

<a id="section-24"></a>

## 24. Paper design and figures

### 24.1 Recommended central claim

A conservative paper framing is:

> We investigate whether the global coefficient structure of pretrained equivariant potentials supports more compact and identifiable subspaces than local weight or radial-function decompositions. We specialize ODT's canonical-environment principle to an exact MACE energy branch, account for symmetry and repeated feature use, and evaluate the resulting subspaces using function-preserving reparameterizations and intervention controls.

Replace “investigate” with an empirical conclusion only after the corresponding experiments have been completed.

### 24.2 Suggested paper organization

**Introduction.** Present the discrepancy between local coordinate structure and composed function structure. Use the HTML's local-image/benchmark-effect separation as motivation while clearly distinguishing benchmark improvement from approximation fidelity.

**Background and source connection.** Explain χ-net unfolding, ODT canonicalization, MACE's repeated-feature contraction, the HTML's exact nullspace, and its radial covector image.

**Objects and method.** Specify the external function space, norm, target branch, quotient, canonical coefficients, mixed-order marginal score, and equivariant projectors. State the valid-cut identity separately from the primary common-projector method.

**Guarantees and limitations.** Give the coefficient error bound, quasioptimality statement, gauge transport, and conversion to energy/force bounds. State where physical-function identifiability is weaker than fixed-lift invariance.

**Experiments.** Exactness gates, rank–fidelity comparisons, gauge/metric robustness, architecture controls, and HTML-inspired intervention/chemical tests.

**Discussion.** Distinguish useful compression, stable representations, and chemical mechanisms. Explain full-model extensions and negative results without inferring universal conclusions.

### 24.3 Figure plan

| Figure | What it should show | What it must not imply |
|---|---|---|
| 1. Source-to-method map | Dooms' tree/canonical environment on one side; the HTML's quotient, radial images, and controls on the other; the exact first-branch construction between them. | That original ODT already supports arbitrary MACE graphs. |
| 2. Interface and target diagram | Primitive radial functions, aggregate features, repeated products, CG/path quotient, linear readout, and separate nonlinear branch. | That every “layer-1” interface has the same gauge or intervention semantics. |
| 3. Local versus composed spectra | Identical tested interfaces, clear norm labels, effective dimensions and tail ranks separately. | That a low effective dimension is the required retained fidelity rank. |
| 4. Rank versus fidelity | Energy differences and force errors for local, global, and random projections with uncertainty. | That coefficient tails alone establish physical accuracy. |
| 5. Gauge transport | Different native coordinates, matching functional images/projectors, invariant spectra where proved. | Invariance to every function-preserving rewrite beyond the tested equivalence class. |
| 6. Mode interaction atlas | Radial images plus reduced-core interactions and body-order difference kernels. | Independent additive energy contributions per mode without an attribution convention. |
| 7. HTML control replication | Local leading-mode content, specified duals, matched random controls, fidelity and reference-error changes. | That an error-improving deletion must be the most important model mechanism. |
| 8. Extension tradeoff | Branch-only versus shared/full-model fidelity, contraction cost, and optional head approximation degree. | Full-model compression from a branch-only result. |

Actual values should be plotted only after computation. Illustrative schematics must be labeled as schematics, and hypothetical compression numbers must never be drawn as measured results.

### 24.4 Claims the title should avoid until established

Avoid “fully interpretable MACE,” “exact decomposition of the entire nonlinear foundation model,” “unique chemical mechanisms,” or “ten-dimensional MACE” unless the evidence genuinely supports those scopes.

Possible eventual titles include *Composed Weight Structure in Equivariant Interatomic Potentials*, *ODT-Style Functional Subspaces in Pretrained MACE*, or *Gauge-Covariant Compression and Interpretation of a MACE Energy Branch*. A stronger title should follow the actual result rather than determine it.

---

<a id="section-25"></a>

## 25. Novelty audit and relation to adjacent work

### 25.1 Mathematical ingredients to attribute rather than claim

Canonical tensor-network forms, whitening, Schmidt/SVD decompositions, hierarchical tensor truncation, representation-theoretic block structure, symmetric polynomial tensors, and inclusion–exclusion are established mathematical ingredients. The first-branch bound is an explicit specialization of commuting-projection/HSVD reasoning to the shared mixed-order objective. Its usefulness here does not establish a new general tensor theorem.

The primary source contribution from Dooms is the χ-net architecture and its efficient ODT organization for the tied binary tree. The primary contribution from the HTML is the specific MACE non-identifiability analysis, functional representations, and empirical interpretation controls. These should remain prominently credited.

### 25.2 What may be novel, subject to literature verification

Potentially distinctive work includes the checkpoint-specific exact compiler, a practical gauge-covariant and symmetry-preserving specialization with controlled tied projections, the distinction between coefficient-lift and physical-function equivalence in this application, and a scientific finding connecting composed subspaces to model-specific interactions under matched controls.

The novelty of those combinations is **not certified by the supplied documents or by the limited external architectural verification performed for this specification**. Before submission, compare with primary work on equivariant tensor-network compression, ACE/MACE basis reduction, symmetric tensor approximation, polynomial-network interpretability, and weight-derived scientific-model interpretation.

### 25.3 A claim-by-claim literature audit

For each proposed contribution, maintain a table with: exact claim; nearest primary paper; shared mathematics; actual difference; required evidence; and whether the distinction survives a strong baseline.

The standard must be stronger than “no paper uses the same project name.” If a prior method already solves the same constrained approximation objective, the contribution may still be an implementation, an application, a new empirical finding, or a sharper identifiability analysis—but it should be described accordingly.

---

<a id="section-26"></a>

## 26. Deliverables and definition of completion

### 26.1 Minimum publishable research package

The minimum complete study consists of:

1. an exact, independently tested representation of the selected MACE first branch;
2. a reproduced and integrated path quotient;
3. a precise weight-derived metric and globally composed decomposition;
4. correct encoder/decoder transport and equivariant projection;
5. measured rank–fidelity comparisons against strong matched baselines;
6. explicit limits on interpretation and full-model scope;
7. reproducible source, checkpoint, configuration, and result artifacts.

A chemical mechanism case study and an actual inference speedup strengthen the paper but are not implied by the minimum package. The full nonlinear-model extension is a separate level of completion.

### 26.2 Result artifact layout

```text
results/<checkpoint_hash>/<measure_hash>/<target>/<method>/
  manifest.json
  interface_registry.json
  coefficient_representation.json
  quotient_diagnostics.json
  integration_diagnostics.json
  spectrum_<species>_<irrep>.npz
  encoder_decoder_<rank>.npz
  mode_cards.jsonl
  coefficient_errors.csv
  fidelity_by_configuration.csv
  reference_errors_by_group.csv
  gauge_tests.json
  body_order_checks.json
  runtime_measurements.csv
  figures/
  exported_evaluator/
```

Every result should be traceable from a figure back to a configuration and an exact checkpoint. Source-reported measurements should live in a separate `source_reported` record and never be mixed with newly measured results.

### 26.3 Present status

This document provides a research design, source mapping, derivations, implementation contracts, experiment plan, and adversarial checks. It does not report a discovered low-rank MACE subspace, a MACE compression ratio, a real-model runtime gain, or a newly identified chemical mechanism.

The accompanying mathematical checks have been executed. The checkpoint-specific implementation and empirical program remain to be carried out.

---

<a id="section-27"></a>

## 27. Source ledger

### 27.1 Primary source D — χ-net and ODT

**Bibliographic identity:** Thomas Dooms, Ward Gauderis, Geraint A. Wiggins, and José Oramas. *Compositionality Unlocks Deep Interpretable Models*. arXiv:2504.02667v1, 3 April 2025. Ten-page source version.

**Supplied artifact:** `chi-nets.txt`, a text extraction of that paper. The original v1 PDF was also inspected, including the architecture, algorithm, spectrum, and truncation figures. Page references below are PDF page numbers starting at 1, not zero-indexed screenshot indices.

| Identifier | Exact location | Use in this specification |
|---|---|---|
| D:S2 | Section 2, p.2 and the symmetrization paragraph at the start of p.3; Eqs.(1)–(2), Fig.1. | Cloning, binary cores, tensor-product lift, tied tree, biases, and local input symmetry. |
| D:S3 | Section 3, p.3. | RQ canonicalization, global Gram, diagonalization, truncation, and the stated error-bound/loss scope. |
| D:S4 | Section 4, pp.3–4, Figs.3–4. | Empirical interpretation, interaction matrices, and near-linear tracing of digit features. |
| D:AA | Appendix A, pp.4–5, Table 1. | Model, dataset, normalization, and training setup. |
| D:AB | Appendix B, p.5, Table 2. | Accuracy comparison with the ReLU baseline. |
| D:AC | Appendix C, p.5, Fig.5. | Source prediction-explanation example. |
| D:AD | Appendix D, pp.5–6, Fig.6. | Additional extracted digit features. |
| D:AE | Appendix E, pp.5–6, Table 3 and Figs.7–9. | Effective dimensions, constant-direction discussion, spectra, loss, and norm removal. |
| D:AF | Appendix F, p.7. | Binary-tree time and memory complexity. |
| D:AG | Appendix G, pp.7–9, Algorithms 1–3 and Figs.10–14. | Exact factor absorption, sibling contraction diagrams, and projector insertion. |
| D:Table3 | Table 3, p.5. | Source SVD/ODT effective L2 bond dimensions. |
| D:Fig2 | Figure 2, p.3. | Empirical accuracy versus dimensions removed. |

Ranges such as `D:S2–S4`, `D:AA–AG`, and `D:AF–AG` refer to the corresponding consecutive source sections. The numerical formulas and proof qualifications in this specification are distinguished from what the source prints; the original wording is not silently treated as a theorem under a different notation.

### 27.2 Primary source H — MACE identifiability and controls

**Identity:** David Olloqui. *Reading the inside of a machine-learned interatomic potential*. Subtitle: *Identifiability, matched controls and a gauge-invariant representation in MACE-OFF23*. Supplied standalone HTML manuscript; no public version identifier or publication status is established by the attachment.

**Supplied artifact:** `shared(1).html`. The references below use its actual anchor IDs and original raw HTML line ranges. Embedded SVG geometry occupies many lines between prose sections; large line jumps do not indicate unread prose. The complete prose, equations, tables, figure captions, appendices, and source-reported numerical record were inspected.

| Identifier | HTML anchor / original lines | Main information used |
|---|---|---|
| H:model | `model`, lines 613–644; `eq-site` Eq.(1). | Two layers, two readouts, energy scaling and branch identity. |
| H:problem | `problem`, lines 1997–2011. | Coordinate freedom and matched-control motivation. |
| H:gauge | `gauge`, lines 3460–3479; `eq-gauge` Eq.(2). | Compensated layer-1 linear maps and scoped gauge family. |
| H:nullspace | `nullspace`, lines 3481–3510; `eq-Tmap` Eq.(3). | Exact symmetrized path nullspace and conversion distinction. |
| H:training | `training`, lines 5554–5569. | Limited small-model training-ensemble evidence. |
| H:detection | `detection`, beginning at line 7628 and the associated control discussion. | Label-table and probe controls. |
| H:selectivity | `selectivity`, beginning at line 9209; `eq-aucid` Eq.(5). | Reach-only support contribution to selectivity. |
| H:steering | `steering`, beginning at line 12646; `eq-afdef` Eq.(6). | Stored coefficient for linear-readout steering. |
| H:synthesis | `synthesis`, lines 14719–14742. | Limits of the three SAE tests and reconstruction resolution. |
| H:curves | `curves`, lines 14766–14792; `eq-curve` and `eq-image`, Eqs.(7)–(8). | Radial functions and invariant covector images. |
| H:comparing | `comparing`, lines 17358–17386; `eq-edit` Eq.(9). | Curve-SVD modes, orthogonal/oblique removal, and duals. |
| H:expansion | `expansion`, lines 17388–17416. | Finite first-branch body order and fragment-extrapolation caveat. |
| H:terms | `terms`, beginning at line 17421 and associated Figure 9 discussion. | Pair and angular kernels and limits of their chemical interpretation. |
| H:selection | `selection`, lines 19868–19886. | Reaction-path benchmark and direction selection. |
| H:alignment | `alignment`, lines 19888–19898. | Functional-image versus coordinate agreement. |
| H:deflation | `deflation`, lines 19901–19933; Table 3 and Figure 10; follow-on discussion around lines 22157–22169. | Leading local modes, the truncation ladder, and matched random subspaces. |
| H:calibrations | `calibrations`, lines 22171–22187. | Removal-rule dependence of benchmark effects. |
| H:limitations | `limitations`, lines 22217–22244, within Section 5. | Scope, coarse ladder resolution, and open interpretation questions. |
| H:prereg | `prereg`, beginning at line 22246. | Preregistered outcomes and retained negative results. |
| H:appA | Appendix A, lines 24118–24215; Eqs.(10)–(15). | MACE message, product, update, and readout formulas. |
| H:appB-coverage | Appendix B, `appB-coverage`, lines 24234–24260. | Non-exhaustive gauge coverage and untested modules. |
| H:appB-training | `appB-training`, lines 24262–24287. | Scope and numerical outcomes of ensemble measurements. |
| H:appC-derivation | `appC-derivation`, beginning at line 24291. | Symmetrized scalar/vector/rank-two path dimensions. |
| H:appC-reduced | `appC-reduced`, lines 24341–24362; `eq-reduced` Eq.(19). | Supported path coordinates and exact forward identity. |
| H:appC-conversion | `appC-conversion`, beginning at line 24364; Table 5 and associated timing discussion. | Converted checkpoints, storage, and inference measurements. |
| H:appD | Appendix D, beginning at line 26389. | Optimization in redundant/reduced coordinates, distinct from forward equivalence. |
| H:appE-resolution | `appE-resolution`, lines 27890–27914. | Energy reconstruction errors and dictionary stability. |
| H:appF | Appendix F, lines 27915–27981. | Reach-only ablation identity and its assumptions. |
| H:appG-expansion | `appG-expansion`, lines 27988–28018; Eqs.(25)–(26). | Möbius expansion, vanishing lemma, and evaluation geometry scope. |
| H:appG-terms | `appG-terms`, lines 28020–28040. | Pair/angle tabulation and force-field fits. |
| H:appG-conformers | `appG-conformers`, lines 28041–28063; Figure 14 and lines 29975–29986; Eq.(27). | Readout-resolved conformer differences and nonlinear cancellation. |
| H:appG-curves | `appG-curves`, lines 29988–30001. | Radial reconstruction across architectures and cross-model curve agreement. |
| H:appG-elements | `appG-elements`, lines 30003–30037. | Element relabeling/chimera and descriptive-geometry control results. |
| H:appH | Appendix H, lines 30040–30151. | Itemized preregistration record and interpretation controls. |
| H:appI | Appendix I, lines 30154–30190, Table 9. | Numerical floors, tolerances, and timing context. |

Line ranges are locators for the supplied snapshot, not a substitute for matching semantic anchors and equations. Source-specific reported numbers remain attributed to this manuscript and are not promoted to independently verified empirical facts.

### 27.3 Design source P

**Artifact:** `Pasted markdown(7).md`.

**Role:** exploratory project concept connecting global weight composition, MACE's polynomial structure, the path quotient, radial images, and a potential full nonlinear extension. It motivates research questions but supplies no experimental verification or additional established theorem. The standalone mathematical definitions and proof scopes are those stated in this specification.

### 27.4 Supplementary primary sources and implementation inspection

| Identifier | Source | Scope of use |
|---|---|---|
| M:arch | Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, arXiv:2206.07697. | Architectural provenance; the HTML explicitly cites its equations. |
| M:off | Kovacs et al., *MACE-OFF: Transferable Short Range Machine Learning Force Fields for Organic Molecules*, arXiv:2312.15211. | Checkpoint-family provenance, not a substitute for the exact loaded model manifest. |
| M:code | Official `ACEsuit/mace` source, notably `mace/modules/blocks.py`, `symmetric_contraction.py`, and `radial.py`, inspected on 21 September 2026. | Linear–activation–linear readout; radial MLPs; multiple interaction variants; channel/product implementation; current reduced-CG option. Inspection was of public main-branch content, not a pinned release used by every source experiment. |
| T:tree | Grelier, Nouy, and Chevreuil, *Learning with Tree-Based Tensor Formats*, arXiv:1811.04455v2; also the HSVD literature cited by Dooms. | Adjacent tensor-format background. The explicit bounds used here are derived in the text under their own assumptions. |
| B:body | Chong et al., *Resolving the Body-Order Paradox of Machine Learning Interatomic Potentials*, arXiv:2509.14146; J. Chem. Phys. 164, 064121 (2026), DOI:10.1063/5.0303302. | Supplementary context on body-order decomposition and its model/data dependence; not new evidence for the specific MACE-OFF23 case studies. |

Reproducible external locators:

```text
D original PDF: https://arxiv.org/pdf/2504.02667v1
M architecture: https://arxiv.org/abs/2206.07697
M OFF family: https://arxiv.org/abs/2312.15211
M code: https://github.com/ACEsuit/mace
M blocks: https://raw.githubusercontent.com/ACEsuit/mace/main/mace/modules/blocks.py
M contractions: https://raw.githubusercontent.com/ACEsuit/mace/main/mace/modules/symmetric_contraction.py
M radial: https://raw.githubusercontent.com/ACEsuit/mace/main/mace/modules/radial.py
Tree formats: https://arxiv.org/abs/1811.04455v2
Body order: https://arxiv.org/abs/2509.14146
```

The supplementary inspection is targeted architectural/mathematical verification, not an exhaustive novelty review. A future implementation must pin exact revisions and check whether source-specific assumptions still hold.

---

<a id="appendix-a"></a>

## Appendix A. Executed mathematical checks

### A.1 Reproduction

The companion script requires Python 3.10 or later and NumPy. It uses a fixed seed and no checkpoint, molecular data, network access, or learned auxiliary model.

```bash
python verify_mace_odt_math.py --output mace_odt_math_checks.json
```

Run without Python's `-O` optimization flag because the verification script uses assertions. Small floating-point differences across numerical libraries are expected; the assertions use explicit tolerances.

The script was run for this specification, and all **16** checks passed. A passing counterexample check means the intended counterexample was exhibited—not that the overbroad claim it tests is true.

### A.2 Check inventory and selected results

| Executed check | What it establishes in the synthetic setting | Recorded result |
|---|---|---|
| `single_bridge_svd` | The small canonical operator reproduces the valid cut's squared singular values and optimal residual. | Spectrum relative error about $5.45\times10^{-16}$; residual squared and discarded eigenvalue sum both about 24.58193817. |
| `bridge_gauge_invariance` | Non-orthogonal gauge transport preserves the valid-cut spectrum, projected function, and covariant projector. | Condition number 20; projector covariance error about $8.02\times10^{-15}$. |
| `local_vs_composed_svd` | Local amplitude can select the wrong direction for the composed operator. | Rank-one local error 1.0 versus composed error 0.01. |
| `mixed_order_shared_projector_bound` | The shared marginal-tail bound dominates actual mixed-order coefficient error on the tested tensors. | Actual squared error about 10.4798; bound about 18.1904. Random competitor comparisons are sanity checks, not a proof of the global optimum. |
| `tree_lift_null_polynomial` | Local tree symmetry need not remove a nonzero lift of the zero polynomial. | Tensor norm about 0.866025; full symmetrization norm 0; largest tested diagonal evaluation about $1.56\times10^{-15}$. |
| `gram_fourth_moment_counterexample` | A small fourth-power tail is not a generic tensor residual certificate. | Gram tail fraction about 0.00249; actual relative tensor error about 0.218, exceeding a requested 0.1. |
| `truncation_does_not_preserve_all_isometries` | Cropping input legs can destroy row isometry. | Original row norm squared 1; projected value 0.5. |
| `density_to_distinct_body_order` | The explicit multiplicities in (32) agree with subset inclusion–exclusion. | Relative error about $3.63\times10^{-16}$. |
| `physical_vs_independent_copy_metric` | Repeated-input moments differ from independent-copy contractions. | Fourth moment 0.2 versus product of second moments about 0.111111. |
| `scalar_terminal_rank_ceiling` | A scalar linear output produces a trivial rank-one terminal Gram. | Rank 1 at hidden dimension 4. |
| `singular_support_transport` | A pseudoinverse gives invariant encoded functions on the reachable support, not a unique covariant extension off it. | Reachable error about $8.52\times10^{-16}$; off-support discrepancy about 0.286. |
| `mixed_order_gauge_transport` | Canonical mixed-order spectra and native projectors transform correctly in a synthetic GL reparameterization. | Spectrum error about $7.19\times10^{-16}$; projector covariance error about $1.73\times10^{-15}$. |
| `semantic_covector_and_decoder_transport` | Covector images and primal–dual pairings use the different required transport laws. | Image error about $1.01\times10^{-16}$; pairing error about $2.22\times10^{-16}$. |
| `energy_and_force_coefficient_bounds` | The energy and derivative bounds hold on a tested symmetric polynomial and coordinate Jacobian. | Energy error about 30.99 bounded by 164.44; gradient error about 91.05 bounded by 836.12. These quantities are synthetic, not eV or physical force measurements. |
| `factorized_marginal_cross_terms` | Factorized cross-path marginal contraction agrees with a dense tensor and detects exact cancellation. | Relative error about $1.92\times10^{-16}$; cancelled tensor norm 0 while the incorrect pathwise squared-norm sum is positive. |
| `literal_source_bound_one_layer_chi` | The source's printed Gram-squared-Frobenius condition, read literally, does not imply its stated tensor-relative tolerance even in the constructed one-layer χ-net. | Both Gram tail fractions about 0.000899 are below the printed threshold 0.003333, but actual relative tensor error is about 0.170664 rather than at most 0.1. See Appendix B. |

These tests complement the algebraic derivations. They cannot establish scalability, real-model support ranks, predictive accuracy, chemical specificity, or the completeness of a gauge family.

---

<a id="appendix-b"></a>

## Appendix B. A concrete audit of the printed ODT truncation condition

### B.1 The distinction being tested

Section 3 of the supplied Dooms v1 paper writes a sufficient condition using differences of squared Frobenius norms of the bond Gram matrices. Under the conventional definition of a Gram matrix, these squared norms sum fourth powers of the underlying singular values. [D:S3, p.3]

The issue is not whether ODT can use correct singular-value-tail bounds. It can, and the specification does so. The issue is whether the printed condition can be used literally without clarifying its notation.

### B.2 One-layer χ-net construction

Take an augmented input $\bar x=(1,x_1,x_2)$, and let

$$
 e=\begin{pmatrix}0&1&0\\0&0&1\end{pmatrix}.
$$

The symmetric bilinear core has only

$$
 Q_{1,1,1}=1,\qquad Q_{2,2,2}=1
$$

nonzero. Thus it computes $(x_1^2,x_2^2)$. It is in the factored bilinear class of the source: take both factor matrices to be the two-dimensional identity. Its output-by-input matricization is row-isometric, as is the embedding.

Use

$$
 u=\operatorname{diag}(1,\sqrt{0.03}).
$$

The output is $(x_1^2,\sqrt{0.03}\,x_2^2)$. The top Gram and either lower-child environment are all

$$
 G=\operatorname{diag}(1,0.03).
$$

Truncate the common lower bond and the top bond to their leading first coordinate. The unfolded depth-one tree contains three projector occurrences: two lower copies and one top projector. This is the source denominator $2^{L+1}-1=3$ at $L=1$.

### B.3 The literal condition passes, but the tensor tolerance fails

For either logical bond,

$$
 \frac{\|G\|_F^2-\|G'\|_F^2}{\|G\|_F^2}
 =\frac{0.03^2}{1+0.03^2}
 \approx0.00089919.
$$

At requested $\epsilon=0.1$, the printed per-bond threshold fraction is

$$
 \frac{\epsilon^2}{3}=0.00333333.
$$

Both bonds satisfy that literal condition. However, the complete coefficient tensor loses its second orthogonal component, so

$$
 \frac{\|\Theta-\Theta'\|_F}{\|\Theta\|_F}
 =\sqrt{\frac{0.03}{1.03}}
 \approx0.170664>0.1.
$$

This is also checked explicitly by constructing the coefficient tensor and the projected tensor in the companion script. The construction does not depend on molecular data or an approximate contraction.

### B.4 Consequence for the project

Use discarded Gram-eigenvalue sums, equivalent to squared singular-value tails, with a separately justified occurrence count or the primary branch's proved shared-projector inequality. Seek clarification of the source's notation before implementing its printed condition as a relative-error certificate.

This audit does not invalidate the source's architecture, canonical environment idea, empirical SVHN observations, or properly formulated HSVD error bounds. It identifies exactly which displayed condition should not be imported without qualification.

---

<a id="appendix-c"></a>

## Appendix C. Compact notation and claim checklist

### C.1 Notation

| Symbol | Meaning |
|---|---|
| $z,z'$ | Central and neighbor species. |
| $\ell,p$ | Angular degree and parity of an irrep. |
| $F_{zz'}^{(\ell)}(r)$ | Native radial-function vector at an explicitly identified interface. |
| $C$ | Supported square-root/orthogonalization factor of the upstream functional metric. |
| $q_z(t)$ | Orthonormal primitive-function vector under the declared structural measure. |
| $b_z(X)$ | Sum of canonical primitive functions over the site's neighbors. |
| $K_{z,\nu}$ | Fully composed symmetric first-branch coefficient tensor of correlation order $\nu$. |
| $\beta_\nu$ | Declared positive coefficient-norm weight for order $\nu$. |
| $\rho_{z,\nu,a}$ | Single-slot reduced coefficient operator. |
| $\Gamma_z$ | Sum of occurrence-weighted reduced operators across orders. |
| $\Pi$ | Orthogonal projector in canonical function coordinates. |
| $D,B$ | Native encoder and decoder maps. |
| $P=BD$ | Native, generally oblique, supported projector. |
| $d_k,b_k$ | Encoder covector and decoder vector for one mode. The latter is distinct from the aggregate $b_z(X)$. |
| $L,R$ | Upstream and downstream Grams for a genuine separator only. |
| $S=C^\top RC$ | Symmetric valid-cut operator with eigenvalues equal to squared singular values. |
| $V_z(S)$ | Distinct-neighbor subset kernel; the argument $S$ is a set here, not the cut operator. |
| $E_L,E_N$ | Linear-readout branch and nonlinear-readout branch of the site or total energy, as explicitly indicated. |

### C.2 Before stating a result

- [ ] The target function, interface, domain, and metric are named.
- [ ] Source-reported numbers are separated from newly measured numbers.
- [ ] A claimed global cut is actually a valid separator, or its alternative environment is specified.
- [ ] Exact path redundancy is not conflated with approximate channel reduction.
- [ ] The native gauge family and analysis-graph transformations are distinguished.
- [ ] Encoders and decoders have the correct primal/dual transformation laws.
- [ ] All repeated occurrences and consumers receive the intended consistent edit.
- [ ] The reported spectral statistic is not mistaken for a fidelity-preserving rank.
- [ ] Branch-only results are not described as full-model compression.
- [ ] Energy offsets, conformational differences, and forces are handled separately.
- [ ] Chemical descriptions survive support, scale, dual, and random-subspace controls.
- [ ] Runtime and storage benefits are measured on the exported evaluator.
- [ ] Negative findings and failed preregistered hypotheses remain in the report.
- [ ] The contribution is stated at the level actually supported by the evidence.

**Final research objective:** determine whether composed checkpoint structure supplies compact, identifiable, physically inspectable functional subspaces in pretrained MACE—and distinguish that possibility from coordinate artifacts, architectural bottlenecks, formal-lift artifacts, and descriptive patterns that do not explain behavior.

**Companion provenance:** `source_manifest.json` records SHA-256 identities of the three supplied sources, this specification, and the executable mathematical-check artifacts. The bundle does not duplicate the source manuscripts.
