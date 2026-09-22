"""Small deterministic mathematical checks accompanying the MACE-ODT specification.

Run: python verify_mace_odt_math.py --output mace_odt_math_checks.json
Requirements: Python 3.10+, NumPy. No MACE checkpoint or molecular data is used.
These tests check finite-dimensional identities, not real-model feasibility.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np


def symmetrize(t: np.ndarray) -> np.ndarray:
    if t.ndim < 1:
        raise ValueError('A coefficient tensor must have at least one input axis.')
    if len(set(t.shape)) != 1:
        raise ValueError('Full slot symmetrization requires equal axis dimensions.')
    return sum(t.transpose(p) for p in itertools.permutations(range(t.ndim))) / math.factorial(t.ndim)


def marginal(t: np.ndarray, axis: int) -> np.ndarray:
    a = np.moveaxis(t, axis, 0).reshape(t.shape[axis], -1)
    return a @ a.T


def project_tensor(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = t.copy()
    for axis in range(t.ndim):
        out = np.moveaxis(np.tensordot(p, out, axes=(1, axis)), 0, axis)
    return out


def eval_tensor(t: np.ndarray, x: np.ndarray) -> float:
    out = t
    for _ in range(t.ndim):
        out = np.tensordot(out, x, axes=(-1, 0))
    return float(out)


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))


def psd_spectrum(a: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh((a+a.T)/2)[::-1]


def run() -> dict:
    rng = np.random.default_rng(260921)
    results = {}
    u = rng.normal(size=(5, 9))
    v = rng.normal(size=(8, 5))
    left, right = u @ u.T, v.T @ v
    c = np.linalg.cholesky(left)
    q = np.linalg.solve(c, u)
    vals, z = np.linalg.eigh(c.T @ right @ c)
    order = np.argsort(vals)[::-1]
    vals, z = vals[order], z[:, order]
    r = 3
    decoder = c @ z[:, :r]
    encoder = np.linalg.solve(c.T, z[:, :r]).T
    p = decoder @ encoder
    full = v @ u
    truncated = v @ p @ u
    squared_s = np.linalg.svd(full, compute_uv=False)[:5] ** 2
    error2 = float(np.linalg.norm(full-truncated)**2)
    tail = float(vals[r:].sum())
    assert relative(vals, squared_s) < 1e-11
    assert abs(error2-tail) < 1e-9
    assert relative(q @ q.T, np.eye(5)) < 1e-11
    assert relative(encoder @ decoder, np.eye(r)) < 1e-11
    results['single_bridge_svd'] = {
        'spectrum_relative_error': relative(vals, squared_s),
        'residual_squared': error2, 'discarded_eigenvalue_sum': tail,
        'projector_idempotence_error': relative(p @ p, p)}

    a, _ = np.linalg.qr(rng.normal(size=(5,5)))
    b, _ = np.linalg.qr(rng.normal(size=(5,5)))
    m = a @ np.diag(np.geomspace(1, 20, 5)) @ b.T
    mi = np.linalg.inv(m)
    ug, vg = m @ u, v @ mi
    cg = np.linalg.cholesky(ug @ ug.T)
    lg, zg = np.linalg.eigh(cg.T @ (vg.T @ vg) @ cg)
    ix = np.argsort(lg)[::-1]; lg, zg = lg[ix], zg[:,ix]
    bg = cg @ zg[:,:r]
    dg = np.linalg.solve(cg.T, zg[:,:r]).T
    pg = bg @ dg
    assert relative(vals, lg) < 1e-10
    assert relative(pg, m @ p @ mi) < 1e-10
    assert relative(vg @ pg @ ug, truncated) < 1e-10
    results['bridge_gauge_invariance'] = {
        'gauge_condition_number': float(np.linalg.cond(m)),
        'spectrum_relative_error': relative(vals, lg),
        'projector_covariance_error': relative(pg, m @ p @ mi),
        'truncated_function_error': relative(vg @ pg @ ug, truncated),
        'raw_upstream_spectrum_change': relative(psd_spectrum(left), psd_spectrum(ug@ug.T))}

    u2 = np.diag([100., 0.01]); v2 = np.diag([0.0001, 100.])
    f = v2 @ u2
    p_local = np.diag([1.,0.]); p_global = np.diag([0.,1.])
    e_local = float(np.linalg.norm(f-v2@p_local@u2))
    e_global = float(np.linalg.norm(f-v2@p_global@u2))
    assert np.isclose(e_local, 1.) and np.isclose(e_global, .01)
    results['local_vs_composed_svd'] = {'local_rank_one_error':e_local,'global_rank_one_error':e_global}

    kernels = {d:symmetrize(rng.normal(size=(5,)*d)) for d in (1,2,3)}
    beta = {1:1.0, 2:.7, 3:.2}
    gamma = sum(beta[d] * sum(marginal(t,j) for j in range(d)) for d,t in kernels.items())
    vals2, vec = np.linalg.eigh(gamma); ix=np.argsort(vals2)[::-1]
    vals2, vec=vals2[ix],vec[:,ix]
    p2 = vec[:,:2] @ vec[:,:2].T
    def err(proj:np.ndarray) -> float:
        return sum(beta[d]*float(np.linalg.norm(t-project_tensor(t,proj))**2) for d,t in kernels.items())
    eps2 = err(p2)
    bound = float(np.trace((np.eye(5)-p2)@gamma))
    assert eps2 <= bound + 1e-10
    ratios=[]
    for _ in range(100):
        o,_=np.linalg.qr(rng.normal(size=(5,2)))
        competitor=err(o@o.T)
        assert eps2 <= 3*competitor + 1e-10
        ratios.append(eps2/competitor)
    results['mixed_order_shared_projector_bound'] = {
        'actual_squared_error':eps2, 'marginal_tail_bound':bound,
        'discarded_eigenvalue_sum':float(vals2[2:].sum()),
        'largest_error_ratio_to_100_competitors':max(ratios),
        'quasioptimality_squared_factor_proved_in_spec':3}

    delta=np.zeros((2,2,2,2))
    delta[0,0,1,1]=delta[1,1,0,0]=.5
    for idx in [(0,1,0,1),(0,1,1,0),(1,0,0,1),(1,0,1,0)]: delta[idx]=-.25
    errs=[abs(eval_tensor(delta,rng.normal(size=2))) for _ in range(100)]
    assert max(errs)<1e-12
    assert np.linalg.norm(delta)>0.1 and np.linalg.norm(symmetrize(delta))<1e-12
    assert relative(delta, delta.transpose(1,0,2,3))<1e-12
    assert relative(delta, delta.transpose(2,3,0,1))<1e-12
    results['tree_lift_null_polynomial'] = {
        'unsymmetrized_tensor_norm':float(np.linalg.norm(delta)),
        'full_symmetrization_norm':float(np.linalg.norm(symmetrize(delta))),
        'max_diagonal_evaluation':max(errs)}

    # A Gram's eigenvalues are squared singular values. Squaring them again
    # is not a generic residual-error certificate for the underlying tensor.
    lam=np.array([1., .05]); target=.1
    gram_tail_ratio=float(lam[1]**2/np.sum(lam**2))
    tensor_error=float(np.sqrt(lam[1]/lam.sum()))
    assert gram_tail_ratio <= target**2 and tensor_error > target
    results['gram_fourth_moment_counterexample'] = {
        'target_relative_tensor_error':target,
        'gram_squared_norm_tail_fraction':gram_tail_ratio,
        'actual_relative_tensor_error':tensor_error,
        'scope':'generic single-cut test; not a full reproduction of the paper theorem'}

    core=np.eye(2)/np.sqrt(2)
    cropped=core[:1,:1]
    assert np.isclose(np.linalg.norm(core)**2,1.)
    assert np.isclose(np.linalg.norm(cropped)**2,.5)
    results['truncation_does_not_preserve_all_isometries'] = {
        'original_one_row_core_norm_squared':float(np.linalg.norm(core)**2),
        'input_projected_core_norm_squared':float(np.linalg.norm(cropped)**2)}

    # Möbius subset kernels for an arbitrary symmetric cubic density polynomial.
    kk={d:symmetrize(rng.normal(size=(3,)*d)) for d in (1,2,3)}
    atoms=rng.normal(size=(3,3))
    def multi(t:np.ndarray, xs:list[np.ndarray]) -> float:
        o=t
        for x in reversed(xs): o=np.tensordot(o,x,axes=(-1,0))
        return float(o)
    def energy(ids:tuple[int,...]) -> float:
        x=atoms[list(ids)].sum(axis=0) if ids else np.zeros(3)
        return sum(eval_tensor(t,x) for t in kk.values())
    def mobius(ids:tuple[int,...])->float:
        ans=0.
        for size in range(len(ids)+1):
            for sub in itertools.combinations(ids,size): ans+=(-1)**(len(ids)-size)*energy(sub)
        return ans
    x,y,z0=atoms
    explicit=[multi(kk[1],[x])+multi(kk[2],[x,x])+multi(kk[3],[x,x,x]),
              2*multi(kk[2],[x,y])+3*multi(kk[3],[x,x,y])+3*multi(kk[3],[x,y,y]),
              6*multi(kk[3],[x,y,z0])]
    measured=[mobius((0,)),mobius((0,1)),mobius((0,1,2))]
    assert relative(np.array(explicit),np.array(measured))<1e-12
    results['density_to_distinct_body_order'] = {'explicit':explicit,'inclusion_exclusion':measured,
        'relative_error':relative(np.array(explicit),np.array(measured))}

    # Input copies evaluated at one physical variable require higher moments.
    fourth=1/5; factorized=(1/3)**2
    assert not np.isclose(fourth,factorized)
    results['physical_vs_independent_copy_metric']={'uniform_x_fourth_moment':fourth,
        'product_of_second_moments':factorized}

    terminal=np.array([[1.,2.,-3.,4.]])
    assert np.linalg.matrix_rank(terminal.T@terminal)==1
    results['scalar_terminal_rank_ceiling']={'rank':1,'hidden_dimension':4}

    # A singular whitening defines functionally unique maps only on its support.
    c0=rng.normal(size=(4,2)); u0=c0@rng.normal(size=(2,6))
    m0=rng.normal(size=(4,4))+4*np.eye(4)
    enc0=np.linalg.pinv(c0); encg=np.linalg.pinv(m0@c0)
    reachable_error=relative(encg@m0@u0,enc0@u0)
    off_support=relative(encg@m0,enc0)
    assert reachable_error<1e-11
    results['singular_support_transport']={'reachable_error':reachable_error,'off_support_non_covariance':off_support}
    # Mixed-order canonical marginals transform orthogonally after a GL gauge.
    features = rng.normal(size=(4, 7))
    native = {nu: symmetrize(rng.normal(size=(4,) * nu)) for nu in (1, 2, 3)}
    transform = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    inv_transform = np.linalg.inv(transform)
    factor = np.linalg.cholesky(features @ features.T)
    gauged_features = transform @ features
    gauged_factor = np.linalg.cholesky(gauged_features @ gauged_features.T)
    canon = {nu: project_tensor(t, factor.T) for nu, t in native.items()}
    gauged_native = {nu: project_tensor(t, inv_transform.T) for nu, t in native.items()}
    gauged_canon = {nu: project_tensor(t, gauged_factor.T) for nu, t in gauged_native.items()}
    gamma = sum(nu * marginal(t, 0) for nu, t in canon.items())
    gamma_g = sum(nu * marginal(t, 0) for nu, t in gauged_canon.items())
    eig, vec = np.linalg.eigh(gamma)
    eig_g, vec_g = np.linalg.eigh(gamma_g)
    chosen, chosen_g = vec[:, -2:], vec_g[:, -2:]
    dec = factor @ chosen
    enc = np.linalg.solve(factor.T, chosen).T
    dec_g = gauged_factor @ chosen_g
    enc_g = np.linalg.solve(gauged_factor.T, chosen_g).T
    projector, projector_g = dec @ enc, dec_g @ enc_g
    spectrum_error = relative(eig, eig_g)
    projector_error = relative(projector_g, transform @ projector @ inv_transform)
    assert spectrum_error < 1e-10
    assert projector_error < 1e-10
    results['mixed_order_gauge_transport'] = {
        'spectrum_relative_error': spectrum_error,
        'projector_covariance_error': projector_error,
        'condition_number': float(np.linalg.cond(transform))}

    # Native encoder covectors and decoder vectors have different transport laws.
    covector = rng.normal(size=4)
    decoder_vector = rng.normal(size=4)
    covector_g = inv_transform.T @ covector
    decoder_vector_g = transform @ decoder_vector
    image_error = relative(covector_g @ gauged_features, covector @ features)
    pairing_error = abs(float(covector_g @ decoder_vector_g - covector @ decoder_vector))
    assert image_error < 1e-11 and pairing_error < 1e-11
    results['semantic_covector_and_decoder_transport'] = {
        'functional_image_error': image_error,
        'dual_pairing_error': pairing_error}

    # Energy and coordinate-gradient coefficient bounds for a symmetric polynomial.
    differences = {nu: symmetrize(rng.normal(size=(4,) * nu)) for nu in (1, 2, 3)}
    beta = {1: 0.8, 2: 1.7, 3: 0.4}
    epsilon = math.sqrt(sum(beta[nu] * np.linalg.norm(t)**2 for nu, t in differences.items()))
    point = rng.normal(size=4)
    jacobian = rng.normal(size=(4, 6))
    value = sum(eval_tensor(t, point) for t in differences.values())
    grad_b = np.zeros(4)
    for nu, t in differences.items():
        contribution = t
        for _ in range(nu - 1):
            contribution = np.tensordot(contribution, point, axes=(-1, 0))
        grad_b += nu * contribution
    grad_x = jacobian.T @ grad_b
    norm_b = np.linalg.norm(point)
    energy_bound = epsilon * math.sqrt(sum(norm_b**(2*nu)/beta[nu] for nu in differences))
    gradient_bound = epsilon * np.linalg.norm(jacobian, 2) * math.sqrt(
        sum(nu**2 * norm_b**(2*nu-2)/beta[nu] for nu in differences))
    assert abs(value) <= energy_bound + 1e-10
    assert np.linalg.norm(grad_x) <= gradient_bound + 1e-10
    results['energy_and_force_coefficient_bounds'] = {
        'absolute_energy_error': abs(float(value)),
        'energy_bound': float(energy_bound),
        'coordinate_gradient_error': float(np.linalg.norm(grad_x)),
        'coordinate_gradient_bound': float(gradient_bound)}

    # CP marginal contraction must retain cross-term interference.
    rank, width, degree = 5, 4, 3
    coeff = rng.normal(size=rank)
    factors = rng.normal(size=(degree, rank, width))
    dense = np.einsum('s,sa,sb,sc->abc', coeff, factors[0], factors[1], factors[2])
    overlaps = (factors[1] @ factors[1].T) * (factors[2] @ factors[2].T)
    factored_marginal = factors[0].T @ (np.outer(coeff, coeff) * overlaps) @ factors[0]
    cp_error = relative(factored_marginal, marginal(dense, 0))
    unit = rng.normal(size=width)
    component = np.einsum('a,b,c->abc', unit, unit, unit)
    cancellation_norm = float(np.linalg.norm(component - component))
    path_norm_sum = float(2*np.linalg.norm(component)**2)
    assert cp_error < 1e-11 and cancellation_norm == 0 and path_norm_sum > 0
    results['factorized_marginal_cross_terms'] = {
        'factorized_vs_dense_error': cp_error,
        'cancelled_tensor_norm': cancellation_norm,
        'incorrect_sum_of_path_squared_norms': path_norm_sum}
    # A complete one-layer chi-net tests the printed Gram-norm condition literally.
    # e selects x1,x2 from (1,x1,x2); Q computes x1^2,x2^2 and is row-isometric.
    # The output has two orthogonal components, with amplitudes 1 and sqrt(.03).
    embedding = np.array([[0., 1., 0.], [0., 0., 1.]])
    quadratic = np.zeros((2, 2, 2))
    quadratic[0, 0, 0] = 1.
    quadratic[1, 1, 1] = 1.
    head = np.diag([1., math.sqrt(0.03)])
    top_gram = head.T @ head
    lower_gram = np.einsum('oab,op,pcb->ac', quadratic, top_gram, quadratic)
    projector = np.diag([1., 0.])
    theta = np.einsum('yo,oab,ai,bj->yij', head, quadratic, embedding, embedding)
    theta_prime = np.einsum('yo,op,pab,ac,bd,ci,dj->yij',
        head, projector, quadratic, projector, projector, embedding, embedding)
    eps = 0.1
    occurrences = 3  # 2^(L+1)-1 at L=1.
    ratios = []
    for gram in (top_gram, lower_gram):
        cropped_gram = projector @ gram @ projector
        ratios.append(float((np.linalg.norm(gram)**2 - np.linalg.norm(cropped_gram)**2)
                            / np.linalg.norm(gram)**2))
    actual_relative_error = float(np.linalg.norm(theta-theta_prime)/np.linalg.norm(theta))
    assert relative(embedding @ embedding.T, np.eye(2)) < 1e-12
    assert relative(quadratic.reshape(2, 4) @ quadratic.reshape(2, 4).T, np.eye(2)) < 1e-12
    assert all(ratio <= eps**2 / occurrences for ratio in ratios)
    assert actual_relative_error > eps
    results['literal_source_bound_one_layer_chi'] = {
        'depth': 1, 'projection_occurrences': occurrences,
        'literal_gram_tail_fractions': ratios,
        'literal_threshold_fraction': eps**2/occurrences,
        'actual_relative_tensor_error': actual_relative_error,
        'requested_relative_tensor_error': eps,
        'interpretation': 'Counterexample to the printed Gram-squared-Frobenius condition read literally; not to correctly formulated HSVD tail bounds.'}
    return {'status':'passed','seed':260921,'test_count':len(results),
        'scope':'finite-dimensional synthetic checks only; no MACE or SVHN experiments', 'tests':results}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('mace_odt_math_checks.json'))
    args=parser.parse_args()
    report=run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
