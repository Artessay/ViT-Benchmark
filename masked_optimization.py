from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn as nn
from torch.optim import Adam


GRADIENT_MASK_ATTRIBUTE = "_gradient_mask"


def _prepare_parameter_masks(
    model: nn.Module,
    parameter_masks: Mapping[str, torch.Tensor],
) -> dict[str, tuple[torch.nn.Parameter, torch.Tensor]]:
    named_parameters = dict(model.named_parameters())
    unknown_names = set(parameter_masks).difference(named_parameters)
    if unknown_names:
        unknown_list = ", ".join(sorted(unknown_names))
        raise KeyError(f"Unknown masked parameters: {unknown_list}")

    prepared_masks = {}
    for name, mask in parameter_masks.items():
        parameter = named_parameters[name]
        if mask.shape != parameter.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match {name} shape {tuple(parameter.shape)}."
            )
        prepared_masks[name] = (
            parameter,
            mask.to(device=parameter.device, dtype=torch.bool).clone(),
        )
    return prepared_masks


def install_gradient_masks(
    model: nn.Module,
    parameter_masks: Mapping[str, torch.Tensor],
) -> None:
    """Install coordinate masks and freeze parameters omitted from the mapping."""
    prepared_masks = _prepare_parameter_masks(model, parameter_masks)
    clear_gradient_masks(model)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter, gradient_mask in prepared_masks.values():
        parameter.requires_grad_(True)
        setattr(parameter, GRADIENT_MASK_ATTRIBUTE, gradient_mask)


@torch.no_grad()
def apply_parameter_masks(
    model: nn.Module,
    parameter_masks: Mapping[str, torch.Tensor],
) -> None:
    """Zero coordinates marked false without changing parameter trainability."""
    prepared_masks = _prepare_parameter_masks(model, parameter_masks)
    for parameter, parameter_mask in prepared_masks.values():
        parameter.mul_(parameter_mask.to(dtype=parameter.dtype))


def has_gradient_masks(model: nn.Module) -> bool:
    """Return whether any model parameter has a coordinate gradient mask."""
    return any(hasattr(parameter, GRADIENT_MASK_ATTRIBUTE) for parameter in model.parameters())


def create_mask_aware_adam(
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> tuple[Adam, list[torch.utils.hooks.RemovableHandle]]:
    """Create Adam while keeping masked coordinates fixed, including weight decay."""
    regular_parameters = []
    masked_parameters = []
    hook_handles = []

    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        gradient_mask = getattr(parameter, GRADIENT_MASK_ATTRIBUTE, None)
        if gradient_mask is None:
            regular_parameters.append(parameter)
            continue

        gradient_mask = gradient_mask.to(device=parameter.device, dtype=torch.bool)

        def mask_gradient(gradient, current_parameter=parameter, current_mask=gradient_mask):
            masked_gradient = torch.where(current_mask, gradient, torch.zeros_like(gradient))
            if weight_decay:
                decayed_gradient = masked_gradient + weight_decay * current_parameter.detach()
                masked_gradient = torch.where(current_mask, decayed_gradient, masked_gradient)
            return masked_gradient

        hook_handles.append(parameter.register_hook(mask_gradient))
        masked_parameters.append(parameter)

    if not masked_parameters:
        optimizer = Adam(regular_parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer, hook_handles

    parameter_groups = []
    if regular_parameters:
        parameter_groups.append({"params": regular_parameters, "weight_decay": weight_decay})
    parameter_groups.append({"params": masked_parameters, "weight_decay": 0.0})
    optimizer = Adam(parameter_groups, lr=learning_rate)
    return optimizer, hook_handles


def clear_gradient_masks(model: nn.Module) -> None:
    """Remove installed coordinate masks from all model parameters."""
    for parameter in model.parameters():
        if hasattr(parameter, GRADIENT_MASK_ATTRIBUTE):
            delattr(parameter, GRADIENT_MASK_ATTRIBUTE)
