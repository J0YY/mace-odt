"""Independent finite-dimensional checks for the attached MACE–ODT specification.

This script does NOT load MACE, a MACE checkpoint, SAE artifacts, or molecular data.
It uses NumPy and the standard library only. It supplements (does not replace)
proofs and a separately rerun copy of the package's 16 bundled checks.

Usage: python independent_checks.py [--output results.json]
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import combinations_with_replacement, permutations, product
import json
import math
from pathlib import Path
import numpy as np


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))


def symmetric(a: np.ndarray) -> np.ndarray:
    ps = list(permutations(range(a.ndim)))
    return sum(a.transpose(p) for p in ps) / len(ps)


def project_all(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = a
    for axis in range(a.ndim):
        out = np.moveaxis(np.tensordot(p, out, axes=(1, axis)), 0, axis)
    return out


def marginal(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.moveaxis(a, axis, 0).reshape(a.shape[axis], -1)
    return m @ m.T


def run() -> dict:
    rng = np.random.default_rng(104729)
    results: dict = {}
    # Representation-theoretic counts: one natural-parity irrep of each l=0,1,2,3.
    # Sym^3 weight multiplicities are counted using monomials in individual basis states.
    states = [(m, (-1) ** l) for l in range(4) for m in range(-l, l + 1)]
    weights = Counter((sum(x[0] for x in terms), math.prod(x[1] for x in terms))
                      for terms in combinations_with_replacement(states, 3))
    counts = []
    for ell, expected_paths, expected_rank in [(0, 23, 8), (1, 51, 12), (2, 65, 14)]:
        parity = (-1) ** ell
        paths = sum(1 for l1, l2, l3 in product(range(4), repeat=3)
                    for intermediate in range(abs(l1-l2), l1+l2+1)
                    if abs(intermediate-l3) <= ell <= intermediate+l3
                    and (-1) ** (l1+l2+l3) == parity)
        rank = weights[ell, parity] - weights[ell+1, parity]
        check((paths, rank) == (expected_paths, expected_rank), 'angular count mismatch')
        counts.append({'ell': ell, 'parity': parity, 'ordered_paths': paths,
                       'symmetric_multiplicity': rank, 'nullity': paths-rank})
    results['angular_counts'] = {'counts': counts,
        'scalar_order_1_paths': 1, 'scalar_order_2_paths': 4,
        'scope': 'Representation-theoretic counts; not a check of serialized MACE buffers.'}

    # The factor-three bound is sharp for mixed orders under the specified norm.
    # K1=sqrt(3-eta)e1, K3=e2^tensor3, rank=1. Objective in t=cos^2(theta)
    # is captured(t)=(3-eta)t+(1-t)^3, a convex function maximized at an endpoint.
    eta = 1e-3
    a2 = 3-eta
    k1 = np.array([np.sqrt(a2), 0.0])
    k3 = np.zeros((2, 2, 2)); k3[1, 1, 1] = 1.0
    gamma = marginal(k1, 0) + sum(marginal(k3, axis) for axis in range(3))
    p_spectral = np.diag([0.0, 1.0]); p_opt = np.diag([1.0, 0.0])
    error = lambda p: sum(float(np.linalg.norm(k-project_all(k, p))**2) for k in [k1, k3])
    check(np.allclose(gamma, np.diag([a2, 3.])), 'sharp example Gamma')
    check(np.isclose(error(p_spectral), a2) and np.isclose(error(p_opt), 1), 'sharp example loss')
    results['sharp_factor_three_example'] = {'eta': eta, 'gamma': gamma.tolist(),
        'spectral_squared_error': error(p_spectral), 'optimal_squared_error': error(p_opt),
        'squared_error_ratio': error(p_spectral)/error(p_opt),
        'norm_error_ratio': float(np.sqrt(error(p_spectral)/error(p_opt)))}

    # Independent randomized stress checks, including nonsymmetric tensors: symmetry
    # is only needed to identify equal marginals, not for the projector sandwich.
    violations = 0; largest_single_order = 0.0; cases = 0
    for dim in (2, 3, 4, 5):
        for rep in range(20):
            tensors = [rng.normal(size=(dim,)*order) for order in range(1, 4)]
            if rep % 2 == 0:
                tensors = [symmetric(k) for k in tensors]
            beta = np.exp(rng.normal(size=3))
            g = sum(b * sum(marginal(k, axis) for axis in range(k.ndim))
                    for b, k in zip(beta, tensors))
            for rank in range(1, dim):
                q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
                p = q @ q.T
                actual = sum(b * np.linalg.norm(k-project_all(k, p))**2
                             for b, k in zip(beta, tensors))
                bound = float(np.trace((np.eye(dim)-p) @ g))
                tolerance = 1e-9 * max(1., actual, abs(bound))
                if not (actual-tolerance <= bound <= 3*actual+tolerance):
                    violations += 1
                largest_single_order = max(largest_single_order, bound/max(actual, 1e-30))
                cases += 1
    check(violations == 0, 'projector sandwich failed')
    results['mixed_order_projector_sandwich'] = {'cases': cases, 'violations': violations,
                                                'largest_bound_over_error': largest_single_order}

    # A spectral tie across a rank cut can select physically different approximants.
    k2 = np.diag([1.0, -1.0]); g = 2*k2@k2.T
    u45 = np.ones(2)/np.sqrt(2); p45 = np.outer(u45, u45)
    e_axis = float(np.linalg.norm(k2-project_all(k2, p_opt))**2)
    e_diagonal = float(np.linalg.norm(k2-project_all(k2, p45))**2)
    check(np.allclose(g, 2*np.eye(2)) and np.isclose(e_axis, 1) and np.isclose(e_diagonal, 2),
          'degenerate example failed')
    results['degenerate_cut'] = {'gamma': g.tolist(), 'rank_one_axis_squared_error': e_axis,
                                'rank_one_diagonal_squared_error': e_diagonal}

    # A two-layer symmetric binary network, independent dense oracle and bottom-up QR.
    embedding = rng.normal(size=(3, 4))
    f1 = rng.normal(size=(3, 3, 3)); f1 = (f1+f1.swapaxes(1, 2))/2
    f2 = rng.normal(size=(2, 3, 3)); f2 = (f2+f2.swapaxes(1, 2))/2
    unembed = rng.normal(size=(2, 2))
    dense = lambda e, a, b, u: np.einsum('oj,jab,acd,bef,cp,dq,er,fs->opqrs',
                                        u, b, a, a, e, e, e, e, optimize=True)
    original = dense(embedding, f1, f2, unembed)
    q0t, r0t = np.linalg.qr(embedding.T, mode='reduced')
    q0, c0 = q0t.T, r0t.T
    a1 = np.einsum('oab,ac,bd->ocd', f1, c0, c0)
    q1t, r1t = np.linalg.qr(a1.reshape(3, -1).T, mode='reduced')
    q1, c1 = q1t.T.reshape(3, 3, 3), r1t.T
    a2 = np.einsum('oab,ac,bd->ocd', f2, c1, c1)
    q2t, r2t = np.linalg.qr(a2.reshape(2, -1).T, mode='reduced')
    q2, c2 = q2t.T.reshape(2, 3, 3), r2t.T
    uo = unembed @ c2
    reparameterized = dense(q0, q1, q2, uo)
    g2 = uo.T@uo
    g1 = np.einsum('oab,op,pdb->ad', q2, g2, q2)
    g0 = np.einsum('oab,op,pdb->ad', q1, g1, q1)
    intermediate_cut = np.einsum('ot,tab,bpq->aopq', uo, q2, q1).reshape(3, -1)
    canonical_tensor = np.einsum('ot,tab,acd,bef->ocdef', uo, q2, q1, q1)
    leaf_cut = np.moveaxis(canonical_tensor, 1, 0).reshape(3, -1)
    errors = {'reparameterized_tensor': relative_error(original, reparameterized),
              'intermediate_environment': relative_error(g1, intermediate_cut@intermediate_cut.T),
              'leaf_environment': relative_error(g0, leaf_cut@leaf_cut.T)}
    check(max(errors.values()) < 1e-11, 'two-layer oracle check failed')
    results['two_layer_binary_network'] = errors

    # Exact first-step Adam counterexample to general nullspace stationarity, no decay.
    # Loss is w1+2w2; its gradient is perpendicular to null(T), T=[1,2].
    gradient = np.array([1., 2.]); null_direction = np.array([2., -1.])/np.sqrt(5)
    step = -0.01*gradient/(np.abs(gradient)+1e-8)
    check(abs(null_direction@gradient) < 1e-12 and abs(null_direction@step) > 1e-4,
          'Adam nullspace counterexample failed')
    results['adaptive_nullspace_counterexample'] = {
        'loss_gradient_kernel_component': float(null_direction@gradient),
        'first_adam_step_kernel_component': float(null_direction@step),
        'weight_decay': 0,
        'scope': 'Generic non-coordinate-aligned nullspace; not a simulation of the specific MACE optimizer.'}

    # Equal FVU does not determine readout reconstruction error.
    sigma = np.eye(2); w = np.array([1., 0.])
    residual_a = np.diag([0.2, 0.]); residual_b = np.diag([0., 0.2])
    fvu_a, fvu_b = np.trace(residual_a)/np.trace(sigma), np.trace(residual_b)/np.trace(sigma)
    check(np.isclose(fvu_a, fvu_b), 'FVU mismatch')
    results['fvu_readout_counterexample'] = {'fvu_both': float(fvu_a),
        'readout_mse_a': float(w@residual_a@w), 'readout_mse_b': float(w@residual_b@w),
        'fvu_times_readout_variance': float(fvu_a*(w@sigma@w))}

    # Gauge at a post-product feature interface with all three consumers compensated.
    p = rng.normal(size=(4, 4)); a = rng.normal(size=(4, 4))
    skips = rng.normal(size=(3, 4, 4)); readout = rng.normal(size=4)
    mixing = np.eye(4) + 0.15*rng.normal(size=(4, 4)); inv = np.linalg.inv(mixing)
    latent = rng.normal(size=(4, 20)); h = p@latent
    hp = (mixing@p)@latent
    errors = {'linear_up_input': relative_error(a@h, (a@inv)@hp),
              'linear_readout': relative_error(readout@h, (readout@inv)@hp),
              'skip_inputs': relative_error(np.einsum('zij,jb->zib', skips, h),
                    np.einsum('zij,jk,kb->zib', skips, inv, hp))}
    check(max(errors.values()) < 1e-11, 'multi-consumer feature gauge failed')
    results['post_product_feature_gauge_algebra'] = errors
    return {'status': 'passed', 'scope': 'Independent algebraic tests; no real-MACE execution.',
            'number_of_test_groups': len(results), 'tests': results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('independent_results.json'))
    args = parser.parse_args()
    results = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
