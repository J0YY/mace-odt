# Algorithm alignment record

## Scope

The implemented target is the first scalar energy branch of MACE-OFF23 small.
It is a bounded order polynomial in the first interaction density. It is not a
deep binary tree and it is not the full MACE potential.

## Mapping to Dooms Algorithms 1 through 3

| Source step | MACE specialization | Implementation | Required check |
|---|---|---|---|
| Algorithm 1, bottom-up orthogonalization | Factor the radial function metric as `L = C C^T` on its supported function space | `radial_interface.supported_metric_factor` | Weighted feature SVD is used directly so the condition number is not squared |
| Algorithm 2, top-down environment | Contract the full composed branch against itself while tracing every sibling slot | `global_environment.native_slot_marginal` | All channel and path cross terms remain inside the contraction |
| Algorithm 3, projector insertion | Use `D = Z^T C+` and `B = C Z` at the registered density interface | `functional_projector.native_functional_map` | Check `D B = I` and apply encoder then decoder |

The Dooms algorithms apply verbatim only when a tensor network cut is a true
separator. The repeated MACE density slots share one logical representation.
They therefore require one tied projector rather than independently truncated
tree bonds.

## Mixed-order specialization

For each central species and order, the compiler produces a symmetric
coefficient tensor `K[order]`. The implemented score is

```text
Gamma = sum over order of beta[order]
        times sum over every slot of rho[order, slot]
```

The same canonical projector acts in every repeated slot. The implementation
checks

```text
actual tied error <= discarded Gamma eigenvalue sum
discarded Gamma eigenvalue sum <= 3 times actual tied error
```

The tail is a sum of Gram eigenvalues. It is not a sum of squared Gram
eigenvalues. This follows the corrected coefficient-error statement in the
research specification rather than treating the printed Dooms Gram-norm
condition as a tensor-error certificate.

## Symmetry restriction

The primitive space is split into complete `ell = 0, 1, 2, 3` irreps. The full
environment must have the form

```text
Gamma = direct sum over ell of Gamma_mult[ell] tensor I[2 ell + 1]
```

The implementation traces magnetic indices and divides by `2 ell + 1` only
after checking the off-block norm and equality of magnetic copies. Projectors
act on multiplicity coordinates and retain every magnetic component.

## Exactness gates already passed

1. The live exact path quotient preserves checkpoint energies and forces.
2. The radial interface reconstructs multineighbor densities and derivatives.
3. The compiled branch matches random features and molecular derivatives.
4. Cubic path-null changes leave the composed coefficients invariant.
5. Dense synthetic kernels agree with the factorized marginal and norm code.
6. The real environment is positive semidefinite within numerical tolerance.
7. Its rotational off-block and magnetic-copy residuals are at the numerical
   floor.
8. Every reported global rank satisfies the tied-error inequalities.

## Claim boundary

The current spectra and coefficient errors concern the first branch under a
declared independent-slot function norm. Molecular rank-fidelity experiments
measure preservation of the frozen model, not DFT accuracy. Full-model
compression requires a shared-interface experiment for every downstream
consumer.
