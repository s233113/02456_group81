import torch
import torch.nn as nn
from typing import Callable

import inspect
import torch_scatter


def segment_softmax(data, segment_ids, eps=1e-7):

    max_values = torch_scatter.scatter_max(data, segment_ids, dim=0)[0]
    max_values = max_values[segment_ids]
    normalized = data - max_values

    numerator = torch.exp(normalized)
    denominator = torch_scatter.scatter_add(numerator, segment_ids, dim=0)
    denominator = denominator[segment_ids]

    softmax = numerator / (denominator + eps)

    return softmax


class PaddedToSegments(nn.Module):
    """Convert a padded tensor with mask to a stacked tensor with segments."""

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        valid_observations = torch.nonzero(mask).squeeze().to(inputs.device)
        collected_values = inputs[mask]
        return collected_values, valid_observations[:, 0]


def cumulative_softmax_weighting(
    values: torch.Tensor,
    preattention: torch.Tensor,
    segment_ids: torch.Tensor,
    eps: float = 1e-7,
):
    """Cumulative softmax weighting of values."""
    n_heads = preattention.size(-1)

    max_values, _ = torch.max(preattention, dim=0, keepdim=True)
    normalized = preattention - max_values
    exp_preattn = torch.exp(normalized)

    cumulative_exp_preattn = torch.zeros_like(exp_preattn)
    weighted_values = torch.zeros(
        *values.shape[:1], n_heads, *values.shape[1:], device=values.device
    )

    for i in range(segment_ids.max().item() + 1):
        mask = segment_ids == i
        cumulative_exp_preattn[mask] = torch.cumsum(exp_preattn[mask], dim=0)
        weighted_values[mask] = values[mask].unsqueeze(1) * exp_preattn[mask].unsqueeze(
            -1
        )

    cumulative_weighted_values = torch.zeros_like(weighted_values)
    for i in range(segment_ids.max().item() + 1):
        mask = segment_ids == i
        cumulative_weighted_values[mask] = torch.cumsum(weighted_values[mask], dim=0)

    out = (cumulative_weighted_values + eps) / (
        cumulative_exp_preattn.unsqueeze(-1) + eps
    )
    return out


class SegmentLayerNormalization(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs: torch.Tensor, segment_ids: torch.Tensor):
        segments, counts = torch.unique(segment_ids, return_counts=True)
        divisor = counts.float() * inputs.size(-1)

        mean = torch.zeros(segments.size(0), device=inputs.device)
        variance = torch.zeros(segments.size(0), device=inputs.device)

        for i, seg in enumerate(segments):
            mask = segment_ids == seg
            seg_data = inputs[mask]
            mean[i] = seg_data.mean()
            variance[i] = seg_data.var(unbiased=False)

        mean = mean[segment_ids]
        variance = variance[segment_ids]

        normalized_inputs = (inputs - mean.unsqueeze(-1)) / torch.sqrt(
            variance.unsqueeze(-1) + self.eps
        )
        return self.gain * normalized_inputs + self.bias


class Segmentpooling(nn.Module):
    def __init__(self, pooling_fn: str = "sum", cumulative: bool = False):
        super().__init__()
        self.cumulative = cumulative
        self.pooling_fn = self._get_pooling_fn(pooling_fn)

    def _get_pooling_fn(self, pooling_fn: str) -> Callable:
        if not self.cumulative:
            if pooling_fn == "sum":
                return lambda x, ids: torch.scatter_add(
                    torch.zeros(ids.max() + 1, *x.shape[1:], device=x.device),
                    0,
                    ids.unsqueeze(-1).expand(-1, x.shape[1]),
                    x,
                )
            elif pooling_fn == "mean":
                return lambda x, ids: torch.scatter_add(
                    torch.zeros(ids.max() + 1, *x.shape[1:], device=x.device),
                    0,
                    ids.unsqueeze(-1).expand(-1, x.shape[1]),
                    x,
                ) / torch.bincount(ids).float().unsqueeze(-1)
            elif pooling_fn == "max":
                return lambda x, ids: torch.scatter_reduce(
                    torch.full(
                        (ids.max() + 1, *x.shape[1:]), float("-inf"), device=x.device
                    ),
                    0,
                    ids.unsqueeze(-1).expand(-1, x.shape[1]),
                    x,
                    reduce="amax",
                )
            else:
                raise ValueError("Invalid pooling function")
        else:
            if pooling_fn == "sum":
                return lambda x, ids: torch.cumsum(x, dim=0)
            elif pooling_fn == "mean":
                return lambda x, ids: torch.cumsum(x, dim=0) / torch.arange(
                    1, x.size(0) + 1, device=x.device
                ).unsqueeze(-1)
            else:
                raise ValueError("Invalid pooling function for cumulative mode")

    def forward(self, data: torch.Tensor, segment_ids: torch.Tensor):
        return self.pooling_fn(data, segment_ids)


# Helper function to map activation names to PyTorch functions
def get_activation_fn(activation_name):
    if activation_name == "relu":
        return nn.ReLU()
    # Add more activation functions as needed
    raise ValueError(f"Unsupported activation function: {activation_name}")


# Custom function to initialize weights
def initialize_weights(layer, initializer_name):
    if initializer_name == "he_uniform":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    # Add more initializers as needed
    else:
        raise ValueError(f"Unsupported initializer: {initializer_name}")


class MySequential(nn.Module):
    def __init__(self, layers):
        super(MySequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, segment_ids=None):
        outputs = inputs  # handle the corner case where self.layers is empty
        for layer in self.layers:
            # Check if the layer's forward method supports 'segment_ids'
            kwargs = {}
            sig = inspect.signature(layer.forward)
            if "segment_ids" in sig.parameters:
                kwargs["segment_ids"] = segment_ids

            # Call the layer with or without segment_ids depending on its signature
            outputs = layer(inputs, **kwargs) if kwargs else layer(inputs)

            # Prepare the input for the next layer
            inputs = outputs

        return outputs


# Updated function to handle activation and kernel_initializer (dense_kwargs)
def build_dense_dropout_model(input_size, n_layers, width, dropout, dense_kwargs):
    """Build a Sequential model composed of stacked Linear and Dropout blocks.

    Args:
        n_layers: Number of layers to stack
        width: Width of the layers
        dropout: Dropout probability
        dense_kwargs: Dictionary for additional layer settings (activation, initializer)

    Returns:
        MySequential model of stacked Linear Dropout layers
    """
    layers = []

    activation_fn = get_activation_fn(dense_kwargs.get("activation", None))
    initializer = dense_kwargs.get("kernel_initializer", None)

    for i in range(n_layers):
        if i == 0:
            linear_layer = nn.Linear(input_size, width)
        else:
            linear_layer = nn.Linear(width, width)

        # Apply kernel initializer if specified
        if initializer:
            initialize_weights(linear_layer, initializer)

        layers.append(linear_layer)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Add activation function if specified
        if activation_fn:
            layers.append(activation_fn)

    return MySequential(layers)
