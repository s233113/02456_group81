import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
import math
from torch.nn.utils.rnn import pad_sequence


from models.seft_utils import (
    build_dense_dropout_model,
    PaddedToSegments,
    Segmentpooling,
    cumulative_softmax_weighting,
    segment_softmax,
)

dense_options = {"activation": "relu", "kernel_initializer": "he_uniform"}


class PositionalEncodingTF(nn.Module):
    """
    Based on the SEFT positional encoding implementation
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        # create a timescale of all times from 0-1
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        # make a tensor to hold the time embeddings
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        # scale the timepoints according to the 0-1 scale
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        # pe = pe.cuda()
        return pe.to(self.device)


class CumulativeSetAttentionLayer(nn.Module):
    def __init__(
        self,
        n_layers: int = 2,
        width: int = 128,
        latent_width: int = 128,
        pooling_function: str = "mean",
        dot_prod_dim: int = 64,
        n_heads: int = 4,
        attn_dropout: float = 0.3,
        psi_input_size: int = 32,
    ):
        super().__init__()
        assert pooling_function == "mean"
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi = build_dense_dropout_model(
            psi_input_size, n_layers, width, 0.0, dense_kwargs=dense_options
        )
        self.psi.layers.append(nn.LazyLinear(latent_width))
        self.rho = nn.LazyLinear(latent_width)
        self.cumulative_segment_mean = Segmentpooling(
            pooling_fn=pooling_function, cumulative=True
        )

    def forward(self, inputs: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.psi(inputs)
        agg = self.cumulative_segment_mean(encoded, segment_ids)
        agg = self.rho(agg)

        combined = torch.cat([inputs, agg], dim=-1)

        keys = self.W_k(combined).view(-1, self.n_heads, 1, self.dot_prod_dim)
        queries = self.W_q.unsqueeze(0).unsqueeze(-1)

        preattn = torch.matmul(keys, queries) / (self.dot_prod_dim**0.5)
        preattn = preattn.squeeze(-1).squeeze(-1)
        return preattn


class SetAttentionLayer(nn.Module):
    def __init__(
        self,
        n_layers: int = 2,
        width: int = 128,
        latent_width: int = 128,
        pooling_function: str = "mean",
        dot_prod_dim: int = 64,
        n_heads: int = 4,
        attn_dropout: float = 0.3,
        psi_input_size: int = 32,
    ):
        super().__init__()
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi_input_size = psi_input_size
        self.psi = build_dense_dropout_model(
            psi_input_size, n_layers, width, 0.0, dense_kwargs=dense_options
        )
        self.psi.layers.append(nn.LazyLinear(latent_width))
        self.psi_pooling = Segmentpooling(pooling_function)
        self.rho = nn.LazyLinear(latent_width)

        self.W_k = nn.Parameter(
            torch.empty(psi_input_size + latent_width, self.dot_prod_dim * self.n_heads)
        )
        nn.init.kaiming_uniform_(self.W_k, a=math.sqrt(5))

        # Weight W_q initialized to zeros
        self.W_q = nn.Parameter(torch.zeros(self.n_heads, self.dot_prod_dim))

    def forward(
        self, inputs: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor
    ) -> List[torch.Tensor]:
        encoded = self.psi(inputs)
        agg = self.psi_pooling(encoded, segment_ids)
        agg = self.rho(agg)
        agg_scattered = agg[segment_ids]
        combined = torch.cat([inputs, agg_scattered], dim=-1)
        keys = torch.matmul(combined, self.W_k).view(
            -1, self.n_heads, 1, self.dot_prod_dim
        )
        queries = self.W_q.unsqueeze(0).unsqueeze((-1))

        preattn = torch.matmul(keys, queries) / (self.dot_prod_dim**0.5)
        preattn = preattn.squeeze(-1)

        if self.training and self.attn_dropout > 0:
            mask = torch.rand_like(preattn) < self.attn_dropout
            preattn = preattn.masked_fill(mask, -1e9)

        return [
            segment_softmax(pre_attn, segment_ids) for pre_attn in preattn.unbind(1)
        ]


class DeepSetAttentionModel(nn.Module):
    def __init__(
        self,
        output_activation,
        output_dims: int,
        seft_n_phi_layers: int,
        seft_phi_width: int,
        seft_n_psi_layers: int,
        seft_psi_width: int,
        seft_psi_latent_width: int,
        seft_dot_prod_dim: int,
        heads: int,
        attn_dropout: float,
        seft_latent_width: int,
        seft_phi_dropout: float,
        seft_n_rho_layers: int,
        seft_rho_width: int,
        seft_rho_dropout: float,
        seft_max_timescales: int,
        seft_n_positional_dims: int,
        n_modalities: int,
        **kwargs
    ):
        super().__init__()

        self.output_activation = (
            torch.nn.Identity()
            if output_activation is None
            else getattr(F, output_activation.lower(), None)
        )

        self.n_modalities = n_modalities

        if n_modalities > 100:
            self.modality_embedding = torch.nn.LazyLinear(64)
            self.n_modalities = 64

        phi_input_dim = self.n_modalities + seft_n_positional_dims + 1

        self.phi = build_dense_dropout_model(
            phi_input_dim,
            seft_n_phi_layers,
            seft_phi_width,
            seft_phi_dropout,
            dense_kwargs=dense_options,
        )
        self.phi.layers.append(nn.LazyLinear(seft_latent_width))
        self.latent_width = seft_latent_width
        self.n_heads = heads

        self.positional_encoding = PositionalEncodingTF(
            d_model=seft_n_positional_dims, max_len=seft_max_timescales
        )

        self.attention = SetAttentionLayer(
            seft_n_psi_layers,
            seft_psi_width,
            seft_psi_latent_width,
            dot_prod_dim=seft_dot_prod_dim,
            n_heads=heads,
            attn_dropout=attn_dropout,
            psi_input_size=phi_input_dim,
        )

        self.pooling = Segmentpooling(pooling_fn="sum", cumulative=False)
        self.demo_encoder = nn.Sequential(
            nn.LazyLinear(seft_phi_width),  # First dense layer
            nn.ReLU(),  # ReLU activation
            nn.LazyLinear(phi_input_dim),  # Second dense layer
        )

        self.rho = build_dense_dropout_model(
            heads * seft_latent_width,
            seft_n_rho_layers,
            seft_rho_width,
            seft_rho_dropout,
            dense_kwargs=dense_options,
        )
        self.rho.layers.append(nn.LazyLinear(output_dims))

        self.to_segments = PaddedToSegments()

    def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:

        x = x.permute(0, 2, 1)
        sensor_mask = sensor_mask.permute(0, 2, 1)

        time, x, sensor_mask, static, lengths = self.flatten_unaligned_measurements(
            x, static, time, sensor_mask
        )

        time = time.squeeze(-1)

        transformed_times = self.positional_encoding(time).squeeze(1)
        transformed_measurements = F.one_hot(sensor_mask, self.n_modalities).float()

        combined_values = torch.cat(
            (transformed_times, x, transformed_measurements), dim=-1
        )

        demo_encoded = self.demo_encoder(static)
        combined_with_demo = torch.cat(
            [demo_encoded.unsqueeze(1), combined_values], dim=1
        )

        if lengths.dim() == 2:
            lengths = lengths.squeeze(-1)

        mask = torch.arange(combined_with_demo.size(1)).unsqueeze(0) < (
            lengths + 1
        ).unsqueeze(1)
        collected_values, segment_ids = self.to_segments(combined_with_demo, mask)

        encoded = self.phi(collected_values)
        attentions = self.attention(collected_values, segment_ids, lengths)

        weighted_values = [encoded * attention for attention in attentions]

        poolingated_values = self.pooling(
            torch.cat(weighted_values, dim=-1), segment_ids
        )
        return self.output_activation(self.rho(poolingated_values))

    def flatten_unaligned_measurements(self, x, static, time, sensor_mask):

        all_gather_y = []
        all_demo = []
        all_gather_x = []
        all_y_indices = []
        all_lengths = []

        for batch_ind in range(x.shape[0]):
            # demo, times, values, measurements, lengths = inputs
            demo = static[batch_ind]
            X = time[batch_ind]
            Y = x[batch_ind]
            measurements = sensor_mask[batch_ind]

            X = X.unsqueeze(-1)
            measurement_positions = torch.nonzero(measurements)
            X_indices = measurement_positions[:, 0]
            Y_indices = measurement_positions[:, 1]

            gathered_X = X[X_indices]
            gathered_Y = Y[
                measurement_positions[:, 0], measurement_positions[:, 1]
            ].unsqueeze(-1)

            length = X_indices.shape[0]

            all_gather_y.append(gathered_Y)
            all_demo.append(demo)
            all_gather_x.append(gathered_X)
            all_y_indices.append(Y_indices)
            all_lengths.append(length)

        # Pad tensors to the same length using torch.nn.utils.rnn.pad_sequence
        padded_gather_y = pad_sequence(
            all_gather_y, batch_first=True, padding_value=0.0
        )  # Y values
        padded_gather_x = pad_sequence(
            all_gather_x, batch_first=True, padding_value=0.0
        )  # X times
        padded_y_indices = pad_sequence(
            all_y_indices, batch_first=True, padding_value=0
        )  # Indices

        # Convert all_demos and all_lengths into tensors
        all_demo = torch.stack(all_demo)  # No padding needed for demo
        all_lengths = torch.tensor(all_lengths)  # Lengths are already a 1D tensor

        # Return the padded tensors and other information
        return padded_gather_x, padded_gather_y, padded_y_indices, all_demo, all_lengths
