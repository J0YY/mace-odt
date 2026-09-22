"""Compile and validate the exact first MACE energy branch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_odt.audit import smoke_geometries, write_json
from mace_odt.first_branch import (
    coefficient_symmetry_residual,
    compile_first_branch,
    compose_order_coefficients,
    evaluate_first_branch_coefficients,
    native_first_branch_node_energy,
)
from mace_odt.path_quotient import compute_path_quotient
from mace_odt.quotient_module import (
    native_weight_for_order,
    torch_symmetrize_repeated_slots,
)


def random_feature_check(model: Any, compiled: Any, seed: int) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    density = torch.randn(
        13,
        compiled.channels,
        compiled.angular_dimension,
        generator=generator,
        dtype=dtype,
    ).to(device)
    indices = torch.arange(13, device=device) % compiled.num_elements
    attrs = torch.nn.functional.one_hot(
        indices, num_classes=compiled.num_elements
    ).to(dtype=dtype)
    with torch.no_grad():
        expected = native_first_branch_node_energy(model, density, attrs)
        observed = evaluate_first_branch_coefficients(compiled, density, attrs)
    difference = observed - expected
    return {
        "max_absolute_error_eV": float(difference.abs().max().cpu()),
        "relative_l2_error": float(
            difference.norm().cpu() / torch.clamp(expected.norm().cpu(), min=1.0)
        ),
    }


def path_null_invariance(model: Any, compiled: Any, seed: int) -> list[dict[str, Any]]:
    contraction = model.products[0].symmetric_contractions.contractions[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records = []
    for order in range(1, 4):
        coupling = torch_symmetrize_repeated_slots(
            contraction.U_tensors(order).detach(), order
        )
        quotient = compute_path_quotient(coupling.cpu().numpy(), order)
        weights = native_weight_for_order(contraction, order).detach()
        perturbed = weights.clone()
        if quotient.null_dimension:
            null_basis = torch.as_tensor(
                quotient.null_basis,
                dtype=weights.dtype,
                device=weights.device,
            )
            coefficients = torch.randn(
                weights.shape[0],
                quotient.null_dimension,
                weights.shape[2],
                generator=generator,
                dtype=weights.dtype,
            ).to(weights.device)
            perturbed = perturbed + torch.einsum(
                "pn,znc->zpc", null_basis, coefficients
            )
        recomposed = compose_order_coefficients(
            coupling, perturbed, compiled.downstream_channel_weights
        )
        difference = recomposed - compiled.orders[order]
        records.append(
            {
                "order": order,
                "path_dimension": quotient.stored_dimension,
                "supported_dimension": quotient.supported_dimension,
                "null_dimension": quotient.null_dimension,
                "coefficient_max_absolute_change": float(
                    difference.abs().max().cpu()
                ),
                "coefficient_relative_change": float(
                    difference.norm().cpu()
                    / torch.clamp(compiled.orders[order].norm().cpu(), min=1.0)
                ),
            }
        )
    return records


def geometry_check(model: Any, calc: Any, compiled: Any, atoms: Any) -> dict[str, Any]:
    from mace.modules.utils import get_edge_vectors_and_lengths

    device = next(model.parameters()).device
    batch = calc._atoms_to_batch(atoms).to(device)
    data = batch.to_dict()
    dtype = next(model.parameters()).dtype
    for key, value in list(data.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            data[key] = value.to(dtype=dtype)
    positions = data["positions"].detach().clone().requires_grad_(True)
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    edge_attrs = model.spherical_harmonics(vectors)
    edge_features, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )
    node_features = model.node_embedding(data["node_attrs"])
    density, residual = model.interactions[0](
        node_attrs=data["node_attrs"],
        node_feats=node_features,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=data["edge_index"],
        cutoff=cutoff,
        first_layer=True,
    )
    if residual is not None:
        raise ValueError("the registered first branch unexpectedly has a residual path")
    native_nodes = native_first_branch_node_energy(model, density, data["node_attrs"])
    compiled_nodes = evaluate_first_branch_coefficients(
        compiled, density, data["node_attrs"]
    )
    native_energy = native_nodes.sum()
    compiled_energy = compiled_nodes.sum()
    native_forces = -torch.autograd.grad(
        native_energy, positions, retain_graph=True
    )[0]
    compiled_forces = -torch.autograd.grad(compiled_energy, positions)[0]
    return {
        "formula": atoms.get_chemical_formula(),
        "node_energy_max_absolute_error_eV": float(
            (compiled_nodes - native_nodes).abs().max().detach().cpu()
        ),
        "total_energy_absolute_error_eV": float(
            (compiled_energy - native_energy).abs().detach().cpu()
        ),
        "force_max_absolute_error_eV_per_A": float(
            (compiled_forces - native_forces).abs().max().detach().cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    compiled = compile_first_branch(model)
    random_check = random_feature_check(model, compiled, seed=20260922)
    path_checks = path_null_invariance(model, compiled, seed=20260923)
    geometry_checks = [
        geometry_check(model, calc, compiled, atoms) for atoms in smoke_geometries()
    ]
    symmetry = {
        str(order): coefficient_symmetry_residual(tensor)
        for order, tensor in compiled.orders.items()
    }
    constant_max = float(compiled.constant.abs().max().cpu())
    observed_errors = [
        random_check["max_absolute_error_eV"],
        constant_max,
        *symmetry.values(),
        *(record["coefficient_max_absolute_change"] for record in path_checks),
        *(record["node_energy_max_absolute_error_eV"] for record in geometry_checks),
        *(record["total_energy_absolute_error_eV"] for record in geometry_checks),
        *(record["force_max_absolute_error_eV_per_A"] for record in geometry_checks),
    ]
    tolerance = 5e-10 if args.dtype == "float64" else 5e-5
    gate_passed = max(observed_errors) <= tolerance

    arrays = {
        "constant_eV": compiled.constant.detach().cpu().numpy(),
        "downstream_channel_weights": (
            compiled.downstream_channel_weights.detach().cpu().numpy()
        ),
        "atomic_numbers": model.atomic_numbers.detach().cpu().numpy(),
    }
    arrays.update(
        {
            f"coefficients_order_{order}": tensor.detach().cpu().numpy()
            for order, tensor in compiled.orders.items()
        }
    )
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **arrays)
    payload = {
        "schema_version": 1,
        "experiment": "E4_exact_first_branch_compiler",
        "requested_model": args.model,
        "dtype": args.dtype,
        "coefficient_axes": {
            "order_1": ["central_species", "channel", "angular_1"],
            "order_2": [
                "central_species",
                "channel",
                "angular_1",
                "angular_2",
            ],
            "order_3": [
                "central_species",
                "channel",
                "angular_1",
                "angular_2",
                "angular_3",
            ],
        },
        "channels": compiled.channels,
        "angular_dimension": compiled.angular_dimension,
        "num_elements": compiled.num_elements,
        "constant_max_absolute_eV": constant_max,
        "symmetry_max_absolute_residuals": symmetry,
        "random_feature_check": random_check,
        "path_null_invariance": path_checks,
        "geometry_checks": geometry_checks,
        "tolerance": tolerance,
        "gate_passed": gate_passed,
        "npz": str(args.npz),
        "scope": (
            "The arrays exactly compile the first scalar readout branch. They do "
            "not include later interaction blocks or atomic reference energies."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.npz}")
    print(f"gate_passed={gate_passed}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
