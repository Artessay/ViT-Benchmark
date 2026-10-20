from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


BID_PATH_POINTS = (0.25, 0.5, 0.75)
BID_FISHER_EPSILON = 1e-12
BID_PROJECTION_CHUNK_SIZE = 256


@dataclass(frozen=True)
class BiDUnitSpec:
    parameter_name: str
    module_name: str
    module: nn.Linear
    unit_axis: int

    @property
    def num_units(self) -> int:
        return self.module.weight.shape[self.unit_axis]


def _validate_uniform_interior_points(path_points: Sequence[float]) -> None:
    if not path_points:
        raise ValueError("BiD-Shapley requires at least one path point.")

    denominator = len(path_points) + 1
    expected_points = [index / denominator for index in range(1, denominator)]
    if any(
        not math.isclose(path_point, expected_point, rel_tol=0.0, abs_tol=1e-12)
        for path_point, expected_point in zip(path_points, expected_points)
    ):
        raise ValueError("BiD-Shapley uses equal-weight points j / (B + 1), ordered by j.")


def collect_bid_unit_specs(model: nn.Module) -> list[BiDUnitSpec]:
    specs = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or "encoder.layers" not in module_name or ".mlp." not in module_name:
            continue

        if module_name.endswith(("mlp.linear_1", "mlp.0")):
            unit_axis = 0
        elif module_name.endswith(("mlp.linear_2", "mlp.3")):
            unit_axis = 1
        else:
            continue

        specs.append(
            BiDUnitSpec(
                parameter_name=f"{module_name}.weight",
                module_name=module_name,
                module=module,
                unit_axis=unit_axis,
            )
        )

    if not specs:
        raise ValueError("BiD-Shapley did not find any supported MLP projection matrices.")
    return specs


def projected_unit_gradients(
    layer_input: torch.Tensor,
    grad_output: torch.Tensor,
    original_weight: torch.Tensor,
    unit_axis: int,
    chunk_size: int = BID_PROJECTION_CHUNK_SIZE,
) -> torch.Tensor:
    if unit_axis not in (0, 1):
        raise ValueError(f"Unsupported unit axis: {unit_axis}")
    if chunk_size <= 0:
        raise ValueError("BiD-Shapley projection chunk size must be positive.")

    projected_chunks = []
    num_units = original_weight.shape[unit_axis]
    for start in range(0, num_units, chunk_size):
        stop = min(start + chunk_size, num_units)
        if unit_axis == 0:
            directional_output = F.linear(layer_input, original_weight[start:stop], bias=None)
            contributions = grad_output[..., start:stop] * directional_output
        else:
            directional_input = torch.matmul(grad_output, original_weight[:, start:stop])
            contributions = layer_input[..., start:stop] * directional_input

        reduction_dims = tuple(range(1, contributions.ndim - 1))
        if reduction_dims:
            contributions = contributions.sum(dim=reduction_dims)
        projected_chunks.append(contributions)

    return torch.cat(projected_chunks, dim=-1)


def projected_focal_unit_gradient(
    layer_input: torch.Tensor,
    grad_output: torch.Tensor,
    original_unit: torch.Tensor,
    unit_index: int,
    unit_axis: int,
) -> torch.Tensor:
    if unit_axis == 0:
        directional_output = torch.matmul(layer_input, original_unit)
        contributions = grad_output.select(-1, unit_index) * directional_output
    elif unit_axis == 1:
        directional_input = torch.matmul(grad_output, original_unit)
        contributions = layer_input.select(-1, unit_index) * directional_input
    else:
        raise ValueError(f"Unsupported unit axis: {unit_axis}")

    reduction_dims = tuple(range(1, contributions.ndim))
    if reduction_dims:
        contributions = contributions.sum(dim=reduction_dims)
    return contributions


def bid_scores_from_statistics(
    mean_projected_gradients: torch.Tensor,
    mean_fisher_row_sums: torch.Tensor,
    baseline_curvature: torch.Tensor,
    path_curvatures: Sequence[torch.Tensor],
    path_points: Sequence[float] = BID_PATH_POINTS,
    epsilon: float = BID_FISHER_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_uniform_interior_points(path_points)
    if len(path_curvatures) != len(path_points):
        raise ValueError("Each BiD-Shapley path point must have one curvature estimate.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("BiD-Shapley Fisher epsilon must be positive.")

    integrated_curvature = torch.zeros_like(baseline_curvature)
    for path_point, curvature in zip(path_points, path_curvatures):
        integrated_curvature += (1.0 - path_point) * curvature
    integrated_curvature *= 2.0 / len(path_points)

    path_weights = integrated_curvature / (baseline_curvature + epsilon)
    cross_curvature = mean_fisher_row_sums - baseline_curvature
    scores = torch.abs(
        -mean_projected_gradients
        + 0.5 * integrated_curvature
        + 0.5 * path_weights * cross_curvature
    )
    return scores, path_weights


@contextmanager
def _temporary_focal_removal_path(
    spec: BiDUnitSpec,
    original_weight: torch.Tensor,
    unit_index: int,
    path_point: float,
) -> Iterator[torch.Tensor]:
    parameter_unit = spec.module.weight.select(spec.unit_axis, unit_index)
    original_unit = original_weight.select(spec.unit_axis, unit_index)
    with torch.no_grad():
        parameter_unit.copy_((1.0 - path_point) * original_unit)
    try:
        yield original_unit
    finally:
        with torch.no_grad():
            parameter_unit.copy_(original_unit)


def compute_bid_shapley_scores(
    model: nn.Module,
    probe_loader: DataLoader,
    device: torch.device,
    path_points: Sequence[float] = BID_PATH_POINTS,
    epsilon: float = BID_FISHER_EPSILON,
) -> Dict[str, torch.Tensor]:
    """Estimate BiD-Shapley scores for ViT MLP projection units.

    The probe objective remains the repository's image-classification cross-entropy.
    Per-example Fisher contractions follow Eq. (8), and every unit uses the focal
    removal path from Eqs. (9) and (12). For the self term, the implementation uses
    the epsilon-stabilized exact-recovery interpretation stated after Eq. (12), then
    adds the within-matrix cross terms from Eq. (7). The paper specifies the three
    interior nodes but not their quadrature coefficients, so this implementation
    explicitly uses an equal-weight interior rule at 0.25, 0.5, and 0.75.

    A literal focal-unit path requires one probe forward/backward pass per unit and
    path point. This is intentionally expensive, but avoids replacing Eq. (9) with
    a mathematically different shared/global removal path.
    """

    _validate_uniform_interior_points(path_points)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("BiD-Shapley Fisher epsilon must be positive.")

    specs = collect_bid_unit_specs(model)
    original_training_states = [
        (module, module.training)
        for module in model.modules()
    ]
    model.to(device)
    original_requires_grad = [
        (parameter, parameter.requires_grad)
        for parameter in model.parameters()
    ]
    original_weights: dict[str, torch.Tensor] = {}
    first_moments: dict[str, torch.Tensor] = {}
    fisher_row_sums: dict[str, torch.Tensor] = {}
    baseline_curvature_sums: dict[str, torch.Tensor] = {}
    path_curvature_sums: list[dict[str, torch.Tensor]] = []
    sample_count = 0

    captures: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    active_parameter_names: set[str] = set()
    handles = []
    progress = None

    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        for spec in specs:
            original_weights[spec.parameter_name] = spec.module.weight.detach().clone()
            first_moments[spec.parameter_name] = torch.zeros(
                spec.num_units,
                dtype=torch.float64,
                device=device,
            )
            fisher_row_sums[spec.parameter_name] = torch.zeros(
                spec.num_units,
                dtype=torch.float64,
                device=device,
            )
            baseline_curvature_sums[spec.parameter_name] = torch.zeros(
                spec.num_units,
                dtype=torch.float64,
                device=device,
            )
        path_curvature_sums = [
            {
                spec.parameter_name: torch.zeros(spec.num_units, dtype=torch.float64, device=device)
                for spec in specs
            }
            for _ in path_points
        ]

        for spec in specs:
            def capture_projection(module, inputs, output, parameter_name=spec.parameter_name):
                if parameter_name in active_parameter_names:
                    captures[parameter_name] = (inputs[0], output)

            handles.append(spec.module.register_forward_hook(capture_projection))

        num_scoring_passes = len(specs) + len(path_points) * sum(spec.num_units for spec in specs)
        try:
            num_probe_batches = len(probe_loader)
        except TypeError:
            num_probe_batches = None
        progress_total = None if num_probe_batches is None else num_probe_batches * num_scoring_passes
        progress = tqdm(total=progress_total, ncols=80, desc="calculating BiD-Shapley values")

        for data in probe_loader:
            images, labels = data[0].to(device), data[1].to(device)

            for spec in specs:
                spec.module.weight.requires_grad_(True)
                active_parameter_names.add(spec.parameter_name)
                captures.clear()
                logits = model(images)
                losses = F.cross_entropy(logits, labels, reduction="none")
                layer_input, projection_output = captures[spec.parameter_name]
                grad_output, = torch.autograd.grad(losses.sum(), projection_output)
                with torch.no_grad():
                    projected_gradients = projected_unit_gradients(
                        layer_input,
                        grad_output,
                        original_weights[spec.parameter_name],
                        spec.unit_axis,
                    ).to(dtype=torch.float64)

                first_moments[spec.parameter_name] += projected_gradients.sum(dim=0)
                fisher_row_sums[spec.parameter_name] += (
                    projected_gradients * projected_gradients.sum(dim=1, keepdim=True)
                ).sum(dim=0)
                baseline_curvature_sums[spec.parameter_name] += projected_gradients.square().sum(dim=0)
                progress.update()
                active_parameter_names.clear()
                captures.clear()
                spec.module.weight.requires_grad_(False)
                del logits, losses, layer_input, projection_output, grad_output, projected_gradients

            sample_count += labels.shape[0]

            for spec in specs:
                spec.module.weight.requires_grad_(True)
                parameter_name = spec.parameter_name
                original_weight = original_weights[parameter_name]
                active_parameter_names.add(parameter_name)

                for unit_index in range(spec.num_units):
                    for path_index, path_point in enumerate(path_points):
                        captures.clear()
                        with _temporary_focal_removal_path(
                            spec,
                            original_weight,
                            unit_index,
                            path_point,
                        ) as original_unit:
                            logits = model(images)
                            losses = F.cross_entropy(logits, labels, reduction="none")
                            layer_input, projection_output = captures[parameter_name]
                            grad_output, = torch.autograd.grad(losses.sum(), projection_output)

                            with torch.no_grad():
                                focal_projected_gradients = projected_focal_unit_gradient(
                                    layer_input,
                                    grad_output,
                                    original_unit,
                                    unit_index,
                                    spec.unit_axis,
                                ).to(dtype=torch.float64)
                                path_curvature_sums[path_index][parameter_name][unit_index] += (
                                    focal_projected_gradients.square().sum()
                                )

                        progress.update()
                        captures.clear()
                        del (
                            logits,
                            losses,
                            layer_input,
                            projection_output,
                            grad_output,
                            focal_projected_gradients,
                        )

                active_parameter_names.clear()
                spec.module.weight.requires_grad_(False)
    finally:
        if progress is not None:
            progress.close()
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for spec in specs:
                if spec.parameter_name in original_weights:
                    spec.module.weight.copy_(original_weights[spec.parameter_name])
        for parameter, requires_grad in original_requires_grad:
            parameter.requires_grad_(requires_grad)
        for module, training in original_training_states:
            module.training = training

    if sample_count == 0:
        raise ValueError("BiD-Shapley requires a non-empty probe loader.")

    scores = {}
    for spec in specs:
        parameter_name = spec.parameter_name
        mean_projected_gradients = first_moments[parameter_name] / sample_count
        mean_fisher_row_sums = fisher_row_sums[parameter_name] / sample_count
        baseline_curvature = baseline_curvature_sums[parameter_name] / sample_count
        path_curvatures = [
            path_curvature_sums[path_index][parameter_name] / sample_count
            for path_index in range(len(path_points))
        ]
        parameter_scores, _ = bid_scores_from_statistics(
            mean_projected_gradients,
            mean_fisher_row_sums,
            baseline_curvature,
            path_curvatures,
            path_points,
            epsilon,
        )
        scores[parameter_name] = parameter_scores.cpu()

    return scores


def select_top_bid_units(
    scores: Mapping[str, torch.Tensor],
    activate_ratio: float,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= activate_ratio <= 1.0:
        raise ValueError("activate_ratio must lie in [0, 1].")

    names = list(scores)
    flattened_scores = torch.cat([scores[name].detach().to(device="cpu", dtype=torch.float64) for name in names])
    num_selected = int(flattened_scores.numel() * activate_ratio)
    selected_flat = torch.zeros(flattened_scores.numel(), dtype=torch.bool)
    if num_selected:
        ranked_indices = torch.argsort(flattened_scores, descending=True, stable=True)
        selected_flat[ranked_indices[:num_selected]] = True

    selected = {}
    offset = 0
    for name in names:
        num_units = scores[name].numel()
        selected[name] = selected_flat[offset : offset + num_units].clone()
        offset += num_units
    return selected


def expand_unit_mask(unit_mask: torch.Tensor, parameter: torch.Tensor, unit_axis: int) -> torch.Tensor:
    if unit_axis == 0:
        return unit_mask.unsqueeze(1).expand_as(parameter)
    if unit_axis == 1:
        return unit_mask.unsqueeze(0).expand_as(parameter)
    raise ValueError(f"Unsupported unit axis: {unit_axis}")


def build_bid_parameter_masks(
    specs: Sequence[BiDUnitSpec],
    selected_units: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Expand BiD row/column selections into full parameter-coordinate masks."""
    parameter_masks = {}
    for spec in specs:
        parameter = spec.module.weight
        if spec.parameter_name not in selected_units:
            raise KeyError(f"Missing selected units for {spec.parameter_name}.")
        unit_mask = selected_units[spec.parameter_name]
        if unit_mask.ndim != 1 or unit_mask.numel() != spec.num_units:
            raise ValueError(
                f"Unit mask for {spec.parameter_name} must have shape ({spec.num_units},)."
            )
        parameter_masks[spec.parameter_name] = expand_unit_mask(
            unit_mask.to(dtype=torch.bool),
            parameter,
            spec.unit_axis,
        ).clone()
    return parameter_masks
