import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from bid_shapley import (
    BID_FISHER_EPSILON,
    BID_PATH_POINTS,
    fisher_linearized_path_curvatures,
    build_bid_parameter_masks,
    collect_bid_unit_specs,
    compute_bid_shapley_scores,
    select_top_bid_units,
)
from masked_optimization import apply_parameter_masks, install_gradient_masks
from vision_transformer import VisionTransformer


def activate_head(model: VisionTransformer):
    for name, param in model.named_parameters():
        if "heads" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def activate_full(model: VisionTransformer):
    for param in model.parameters():
        param.requires_grad = True


def activate_random(model: VisionTransformer, activate_ratio: float):
    """randomly select some neurals, and froze other neurals"""
    neuron_groups = {}
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            # Get the neuron identifier by removing the .weight or .bias suffix
            neuron_key = name.rsplit(".", 1)[0]
            if neuron_key not in neuron_groups:
                neuron_groups[neuron_key] = []
            neuron_groups[neuron_key].append((name, param))
        else:
            param.requires_grad = True

    # Randomly select neuron groups to activate
    num_neurons = len(neuron_groups)
    num_activate = int(num_neurons * activate_ratio)
    activate_neuron_keys = random.sample(list(neuron_groups.keys()), num_activate)

    # Set the requires_grad attribute of parameters
    for neuron_key, param_group in neuron_groups.items():
        if neuron_key in activate_neuron_keys:
            for _, param in param_group:
                param.requires_grad = True
        else:
            for _, param in param_group:
                param.requires_grad = False


def activate_neuron_random(model: VisionTransformer, activate_ratio: float):
    """randomly select some neurals, and froze other neurals"""
    neuron_groups = {}
    for name, param in model.named_parameters():
        if "encoder.layers" in name and "mlp" in name:
            # Get the neuron identifier by removing the .weight or .bias suffix
            neuron_key = name.rsplit(".", 1)[0]
            neuron_value = name.rsplit(".", 1)[1]
            assert neuron_value in ["weight", "bias"]

            if neuron_key not in neuron_groups:
                neuron_groups[neuron_key] = {}
            neuron_groups[neuron_key][neuron_value] = param

    # Set the requires_grad attribute of parameters
    for neuron_key, param_dict in neuron_groups.items():
        weight_param: torch.nn.Parameter = param_dict["weight"]
        bias_param: torch.nn.Parameter = param_dict["bias"]
        assert weight_param.shape[0] == bias_param.shape[0]
        num_neurons = weight_param.shape[0]
        num_activate = int(num_neurons * activate_ratio)

        # Randomly select neuron groups to activate
        activate_indices = random.sample(range(num_neurons), k=num_activate)

        # Create a mask to indicate activate neurons
        mask = torch.zeros(num_neurons, dtype=torch.bool)
        mask[activate_indices] = True

        # Set non - activate neurons' weight and bias to 0
        weight_param.data[~mask] = 0
        bias_param.data[~mask] = 0


def activate_based_on_gradient(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    """
    Activate neurons in the model based on the gradient.

    Args:
        model (VisionTransformer): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    model.to(device)
    model.eval()

    # Store the gradient trace of each neuron
    neuron_gradient = {}

    for data in tqdm(val_loader, ncols=80, desc="calculating gradient"):
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to compute gradients
        loss.backward()

        # Compute gradient traces
        for name, param in model.named_parameters():
            if "encoder.layers" in name:
                # Remove the .weight or .bias suffix to get the neuron identifier
                neuron_key = name.rsplit(".", 1)[0]
                if neuron_key not in neuron_gradient:
                    neuron_gradient[neuron_key] = 0

                # Compute the gradient trace
                if param.grad is not None:
                    param_gradient = torch.abs(param.grad).sum().item()
                    neuron_gradient[neuron_key] += param_gradient

    # Sort neurons by gradient trace
    sorted_neuron_keys = sorted(neuron_gradient.items(), key=lambda item: item[1], reverse=True)

    # Select the number of neurons to activate
    num_neurons = len(sorted_neuron_keys)
    num_activate = int(num_neurons * activate_ratio)
    activate_neuron_keys = [key for key, _ in sorted_neuron_keys[:num_activate]]

    # Set the requires_grad attribute of parameters
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            neuron_key = name.rsplit(".", 1)[0]
            if neuron_key in activate_neuron_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True


def activate_neuron_based_on_gradient(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    """
    Activate neurons in the model based on the gradient.

    Args:
        model (VisionTransformer): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    model.to(device)
    model.eval()

    # Store the gradient trace of each weight
    weight_neuron_gradient = {}

    for data in tqdm(val_loader, ncols=80, desc="Calculating gradient"):
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to compute gradients
        loss.backward()

        # Compute gradient traces only for weights
        for name, param in model.named_parameters():
            if "encoder.layers" in name and "mlp" in name and "weight" in name:
                assert param.grad is not None

                # Compute the gradient
                param_gradient = torch.abs(param.grad).detach()
                if name not in weight_neuron_gradient:
                    weight_neuron_gradient[name] = param_gradient
                else:
                    weight_neuron_gradient[name] += param_gradient

    # Activate neurons based on gradient traces
    for name, param in model.named_parameters():
        if "encoder.layers" in name and "mlp" in name:
            if "weight" in name:
                param_gradient = weight_neuron_gradient[name]
                output_dim = param_gradient.shape[0]
                num_to_activate = int(output_dim * activate_ratio)
                # Calculate the sum of gradient traces for each output dimension
                trace_per_output = param_gradient.sum(dim=1)
                sorted_indices = torch.argsort(trace_per_output, descending=True)
                top_indices = sorted_indices[:num_to_activate]
                mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
                mask[top_indices] = True
                # Expand the mask to the shape of the weight
                expanded_mask = mask.unsqueeze(1).expand_as(param)
                param.data *= expanded_mask.float()
            elif "bias" in name:
                # Find the corresponding weight name
                weight_name = name.rsplit(".", 1)[0] + ".weight"
                param_gradient = weight_neuron_gradient[weight_name]
                output_dim = param_gradient.shape[0]
                num_to_activate = int(output_dim * activate_ratio)
                # Calculate the sum of gradient traces for each output dimension
                trace_per_output = param_gradient.sum(dim=1)
                sorted_indices = torch.argsort(trace_per_output, descending=True)
                top_indices = sorted_indices[:num_to_activate]
                mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
                mask[top_indices] = True
                param.data *= mask.float()


def activate_based_on_gradient_trace(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    """
    Activate neurons in the model based on the gradient trace.

    .. math::
        I(w_{i}) = |w_{i} \\nabla_{w_{i}} \\mathcal{L}|

    Args:
        model (VisionTransformer): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    model.to(device)
    model.eval()

    # Store the gradient trace of each neuron
    neuron_gradient_traces = {}

    for data in tqdm(val_loader, ncols=80, desc="calculating gradient traces"):
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to compute gradients
        loss.backward()

        # Compute gradient traces
        for name, param in model.named_parameters():
            if "encoder.layers" in name:
                # Remove the .weight or .bias suffix to get the neuron identifier
                neuron_key = name.rsplit(".", 1)[0]
                if neuron_key not in neuron_gradient_traces:
                    neuron_gradient_traces[neuron_key] = 0

                # Compute the gradient trace
                if param.grad is not None:
                    gradient_trace = torch.abs(param * param.grad).sum().item()
                    neuron_gradient_traces[neuron_key] += gradient_trace

    # Sort neurons by gradient trace
    sorted_neuron_keys = sorted(neuron_gradient_traces.items(), key=lambda item: item[1], reverse=True)

    # Select the number of neurons to activate
    num_neurons = len(sorted_neuron_keys)
    num_activate = int(num_neurons * activate_ratio)
    activate_neuron_keys = [key for key, _ in sorted_neuron_keys[:num_activate]]

    # Set the requires_grad attribute of parameters
    for name, param in model.named_parameters():
        if "encoder.layers" in name:
            neuron_key = name.rsplit(".", 1)[0]
            if neuron_key in activate_neuron_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True


def activate_neuron_based_on_gradient_trace(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    """
    Activate neurons in the model based on the gradient trace.

    .. math::
        I(w_{i}) = |w_{i} \\nabla_{w_{i}} \\mathcal{L}|

    Args:
        model (VisionTransformer): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    model.to(device)
    model.eval()

    # Store the gradient trace of each weight
    weight_gradient_traces = {}

    for data in tqdm(val_loader, ncols=80, desc="Calculating gradient traces"):
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to compute gradients
        loss.backward()

        # Compute gradient traces only for weights
        for name, param in model.named_parameters():
            if "encoder.layers" in name and "mlp" in name and "weight" in name:
                assert param.grad is not None

                # Compute the gradient trace
                with torch.no_grad():
                    gradient_trace = torch.abs(param * param.grad).detach()
                    if name not in weight_gradient_traces:
                        weight_gradient_traces[name] = gradient_trace
                    else:
                        weight_gradient_traces[name] += gradient_trace

    # Activate neurons based on gradient traces
    for name, param in model.named_parameters():
        if "encoder.layers" in name and "mlp" in name:
            if "weight" in name:
                gradient_trace = weight_gradient_traces[name]
                output_dim = gradient_trace.shape[0]
                num_to_activate = int(output_dim * activate_ratio)
                # Calculate the sum of gradient traces for each output dimension
                trace_per_output = gradient_trace.sum(dim=1)
                sorted_indices = torch.argsort(trace_per_output, descending=True)
                top_indices = sorted_indices[:num_to_activate]
                mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
                mask[top_indices] = True
                # Expand the mask to the shape of the weight
                expanded_mask = mask.unsqueeze(1).expand_as(param)
                param.data *= expanded_mask.float()
            elif "bias" in name:
                # Find the corresponding weight name
                weight_name = name.rsplit(".", 1)[0] + ".weight"
                gradient_trace = weight_gradient_traces[weight_name]
                output_dim = gradient_trace.shape[0]
                num_to_activate = int(output_dim * activate_ratio)
                # Calculate the sum of gradient traces for each output dimension
                trace_per_output = gradient_trace.sum(dim=1)
                sorted_indices = torch.argsort(trace_per_output, descending=True)
                top_indices = sorted_indices[:num_to_activate]
                mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
                mask[top_indices] = True
                param.data *= mask.float()


@torch.no_grad()
def calculate_shapley_value(param: torch.nn.Parameter) -> torch.Tensor:
    """Original s-ncft score with fixed interaction coefficient 0.5."""
    return _calculate_shapley_value(param, path_integral=False)


@torch.no_grad()
def calculate_dynamic_shapley_value(param: torch.nn.Parameter) -> torch.Tensor:
    """d-ncft: change only the interaction coefficient in the s-ncft score."""
    return _calculate_shapley_value(param, path_integral=True)


@torch.no_grad()
def calculate_path_interaction_coefficients(param: torch.nn.Parameter) -> torch.Tensor:
    """Compute one normalized path coefficient per output row.

    Use s-ncft's batch-mean gradient, q_i = <grad_i, weight_i>, and the
    directional empirical-Fisher proxy C_i(0) = q_i**2. As in b-ncft,
    Fisher-linearized transport gives q_i(s) ~= q_i - s C_i(0).
    With B=8 interior points, a_i = mean_j[(1-s_j) C_i(s_j)/C_i(0)].
    Constant path curvature recovers a_i=0.5. A degenerate direction with
    C_i(0) <= epsilon falls back to 0.5 rather than erasing interactions.

    This is a local path approximation, not perturbed-model reevaluation.
    In particular it intentionally retains batch gradients instead of b-ncft's
    per-example statistics to isolate the coefficient change from s-ncft.
    """
    assert param.grad is not None
    assert param.ndim == 2
    projected_gradient = (param.grad * param).sum(dim=1)
    baseline_curvature = projected_gradient.square()
    path_curvatures = fisher_linearized_path_curvatures(
        projected_gradient, baseline_curvature, BID_PATH_POINTS
    )
    integral = sum(
        (1.0 - point) * curvature
        for point, curvature in zip(BID_PATH_POINTS, path_curvatures)
    ) / len(BID_PATH_POINTS)
    return torch.where(
        baseline_curvature > BID_FISHER_EPSILON,
        integral / baseline_curvature.clamp_min(BID_FISHER_EPSILON),
        torch.full_like(baseline_curvature, 0.5),
    )


@torch.no_grad()
def _calculate_shapley_value(param: torch.nn.Parameter, *, path_integral: bool) -> torch.Tensor:
    assert param.grad is not None
    assert param.ndim == 2, "Shapley value calculation now is only supported for 2D parameters."
    hessian_matrix = torch.matmul(param.grad, param.grad.T)
    individual_importance = -torch.sum(param.grad * param, dim=1)
    coefficient = calculate_path_interaction_coefficients(param) if path_integral else 0.5
    cooperative_interactions = -coefficient * torch.sum(
        param * torch.matmul(hessian_matrix, param), dim=1
    )
    return individual_importance + cooperative_interactions


def activate_based_on_shapley_value(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    return _activate_based_on_shapley_value(model, activate_ratio, val_loader, device, calculate_shapley_value)


def activate_based_on_dynamic_shapley_value(model, activate_ratio, val_loader, device):
    return _activate_based_on_shapley_value(model, activate_ratio, val_loader, device, calculate_dynamic_shapley_value)


def _activate_based_on_shapley_value(model, activate_ratio, val_loader, device, score_function):
    """
    Activate neurons in the model based on the Shapley value.

    Args:
        model (VisionTransformer): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    model.to(device)
    model.eval()

    def param_name_check(name: str) -> bool:
        """
        Check if the parameter name is valid for Shapley value calculation.

        Args:
            name (str): The name of the parameter.

        Returns:
            bool: True if the parameter name is valid, False otherwise.
        """
        return "encoder.layers" in name and "mlp" in name and "weight" in name

    # Store the gradient trace of each neuron
    neuron_shapley_values = {}

    for data in tqdm(val_loader, ncols=80, desc="calculating shapley values"):
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to compute gradients
        loss.backward()

        # Compute gradient traces
        for name, param in model.named_parameters():
            if param_name_check(name):
                # Remove the .weight or .bias suffix to get the neuron identifier
                neuron_key = name.rsplit(".", 1)[0]

                if neuron_key not in neuron_shapley_values:
                    neuron_shapley_values[neuron_key] = 0

                # Compute the Shapley value for the parameter
                param_shapley_value = score_function(param)
                shapley_value = param_shapley_value.sum().item()

                neuron_shapley_values[neuron_key] += shapley_value

    # Sort neurons by gradient trace
    sorted_neuron_keys = sorted(neuron_shapley_values.items(), key=lambda item: item[1], reverse=True)

    # Select the number of neurons to activate
    num_neurons = len(sorted_neuron_keys)
    num_activate = int(num_neurons * activate_ratio)
    activate_neuron_keys = [key for key, _ in sorted_neuron_keys[:num_activate]]

    # Set the requires_grad attribute of parameters
    for name, param in model.named_parameters():
        if param_name_check(name):
            neuron_key = name.rsplit(".", 1)[0]
            if neuron_key in activate_neuron_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True


def activate_neuron_based_on_shapley_value(model: VisionTransformer, activate_ratio: float, val_loader: DataLoader, device):
    return _activate_neuron_based_on_shapley_value(model, activate_ratio, val_loader, device, calculate_shapley_value)


def activate_neuron_based_on_dynamic_shapley_value(model, activate_ratio, val_loader, device):
    return _activate_neuron_based_on_shapley_value(model, activate_ratio, val_loader, device, calculate_dynamic_shapley_value)


def _activate_neuron_based_on_shapley_value(model, activate_ratio, val_loader, device, score_function):
    """
    Activate neurons in the model based on the Shapley value.

    Args:
        model (VisionTransformer): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        device: The device (CPU or GPU) to run the model on.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    model.to(device)
    model.eval()

    def param_name_check(name: str) -> bool:
        """
        Check if the parameter name is valid for Shapley value calculation.

        Args:
            name (str): The name of the parameter.

        Returns:
            bool: True if the parameter name is valid, False otherwise.
        """
        return "encoder.layers" in name and "mlp" in name and "weight" in name

    # Store the gradient trace of each neuron
    weight_shapley_values = {}

    for data in tqdm(val_loader, ncols=80, desc="calculating shapley values"):
        images, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients
        model.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to compute gradients
        loss.backward()

        # Compute gradient traces
        for name, param in model.named_parameters():
            if param_name_check(name):
                # Compute the Shapley value for the parameter
                param_shapley_value = score_function(param).detach()
                param_shapley_value = torch.abs(param_shapley_value)

                if name not in weight_shapley_values:
                    weight_shapley_values[name] = param_shapley_value
                else:
                    weight_shapley_values[name] += param_shapley_value

    # Activate neurons based on gradient traces
    for name, param in model.named_parameters():
        if "encoder.layers" in name and "mlp" in name:
            if "weight" in name:
                shapley_value = weight_shapley_values[name]
                output_dim = shapley_value.shape[0]
                num_to_activate = int(output_dim * activate_ratio)

                # select the top num_to_activate neurons
                sorted_indices = torch.argsort(shapley_value, descending=True)
                top_indices = sorted_indices[:num_to_activate]
                mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
                mask[top_indices] = True

                # Expand the mask to the shape of the weight
                expanded_mask = mask.unsqueeze(1).expand_as(param)
                param.data *= expanded_mask.float()
            elif "bias" in name:
                # Find the corresponding weight name
                weight_name = name.rsplit(".", 1)[0] + ".weight"
                shapley_value = weight_shapley_values[weight_name]
                output_dim = shapley_value.shape[0]
                num_to_activate = int(output_dim * activate_ratio)

                # select the top num_to_activate neurons
                sorted_indices = torch.argsort(shapley_value, descending=True)
                top_indices = sorted_indices[:num_to_activate]
                mask = torch.zeros(output_dim, dtype=torch.bool, device=device)
                mask[top_indices] = True
                param.data *= mask.float()


def activate_based_on_bid_shapley_value(
    model: VisionTransformer,
    activate_ratio: float,
    val_loader: DataLoader,
    device,
):
    scores = compute_bid_shapley_scores(model, val_loader, device)
    selected_units = select_top_bid_units(scores, activate_ratio)
    specs = collect_bid_unit_specs(model)
    parameter_masks = build_bid_parameter_masks(specs, selected_units)
    install_gradient_masks(model, parameter_masks)


def activate_neuron_based_on_bid_shapley_value(
    model: VisionTransformer,
    activate_ratio: float,
    val_loader: DataLoader,
    device,
):
    scores = compute_bid_shapley_scores(model, val_loader, device)
    selected_units = select_top_bid_units(scores, activate_ratio)
    specs = collect_bid_unit_specs(model)
    parameter_masks = build_bid_parameter_masks(specs, selected_units)
    apply_parameter_masks(model, parameter_masks)
