"""Embedding layers for the models."""

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers import BigBirdConfig, MambaConfig


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> Any:
        """Apply time embedding to the input time stamps."""
        if self.is_time_delta:
            # If the time_stamps represent time deltas, we calculate the deltas.
            # This is equivalent to the difference between consecutive elements.
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi

        return torch.sin(next_input)


# class VisitEmbedding(nn.Module):
#     """Embedding layer for visit segments."""

#     def __init__(
#         self,
#         visit_order_size: int,
#         embedding_size: int,
#     ):
#         super().__init__()
#         self.visit_order_size = visit_order_size
#         self.embedding_size = embedding_size
#         self.embedding = nn.Embedding(self.visit_order_size, self.embedding_size)

#     def forward(self, visit_segments: torch.Tensor) -> Any:
#         """Apply visit embedding to the input visit segments."""
#         return self.embedding(visit_segments)


# class ConceptEmbedding(nn.Module):
#     """Embedding layer for event concepts."""

#     def __init__(
#         self,
#         num_embeddings: int,
#         embedding_size: int,
#         padding_idx: Optional[int] = None,
#     ):
#         super(ConceptEmbedding, self).__init__()
#         self.embedding = nn.Embedding(
#             num_embeddings,
#             embedding_size,
#             padding_idx=padding_idx,
#         )

#     def forward(self, inputs: torch.Tensor) -> Any:
#         """Apply concept embedding to the input concepts."""
#         return self.embedding(inputs)


# class PositionalEmbedding(nn.Module):
#     """Positional embedding layer."""

#     def __init__(self, embedding_size: int, max_len: int):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, embedding_size).float()
#         pe.require_grad = False

#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (
#             torch.arange(0, embedding_size, 2).float()
#             * -(math.log(10000.0) / embedding_size)
#         ).exp()

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         self.register_buffer("pe", pe)

#     def forward(self, visit_orders: torch.Tensor) -> Any:
#         """Apply positional embedding to the input visit orders."""
#         first_visit_concept_orders = visit_orders[:, 0:1]
#         normalized_visit_orders = torch.clamp(
#             visit_orders - first_visit_concept_orders,
#             0,
#             self.pe.size(0) - 1,
#         )
#         return self.pe[normalized_visit_orders]


class MambaEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from concept, token_type, etc., embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: MambaConfig,
        #static:
        max_static_size: int = 8,
        # for ts_indicators:
        max_indicators: int =37,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        #change this for ts_times (1, 215)
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """Initiate wrapper class for embeddings used in Mamba CEHR classes."""
        super().__init__()
        # self.type_vocab_size = type_vocab_size
        self.max_static_size = max_static_size
        self.max_indicators = max_indicators
        self.max_num_visits = max_num_visits
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = config.hidden_size

        # self.word_embeddings = nn.Embedding(
        #     config.vocab_size,
        #     config.hidden_size,
        #     padding_idx=config.pad_token_id,
        # )
        # self.token_type_embeddings = nn.Embedding(
        #     self.type_vocab_size,
        #     config.hidden_size,
        # )

        #Consider if it makes sense to embed it as a time series
        self.indicators_embeddings = nn.Embedding(
            self.type_vocab_size,
            config.hidden_size,
        )
        # self.visit_order_embeddings = nn.Embedding(
        #     self.max_num_visits,
        #     config.hidden_size,
        # )
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        # self.age_embeddings = TimeEmbeddingLayer(
        #     embedding_size=time_embeddings_size,
        # )
        # self.visit_segment_embeddings = VisitEmbedding(
        #     visit_order_size=visit_order_size,
        #     embedding_size=config.hidden_size,
        # )
        self.scale_back_concat_layer = nn.Linear(
            config.hidden_size + 2 * time_embeddings_size,
            config.hidden_size,
        )
        self.static_embeddings = nn.Embedding(
            self.max_static_size,
            config.hidden_size,
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # End copy

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids_batch: Optional[torch.Tensor] = None,
        time_stamps: Optional[torch.Tensor] = None,
        ages: Optional[torch.Tensor] = None,
        visit_orders: Optional[torch.Tensor] = None,
        visit_segments: Optional[torch.Tensor] = None,
    ) -> Any:
        """Return the final embeddings of concept ids.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input data (concept_ids) to be embedded.
        inputs_embeds : torch.Tensor
            The embeddings of the input data.
        token_type_ids_batch : torch.Tensor
            The token type IDs of the input data.
        time_stamps : torch.Tensor
            Time stamps of the input data.
        ages : torch.Tensor
            Ages of the input data.
        visit_orders : torch.Tensor
            Visit orders of the input data.
        visit_segments : torch.Tensor
            Visit segments of the input data.
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Using cached values from a prior cache_input call
        time_stamps_embeds = self.time_embeddings(time_stamps)
        ages_embeds = self.age_embeddings(ages)
        visit_segments_embeds = self.visit_segment_embeddings(visit_segments)
        visit_order_embeds = self.visit_order_embeddings(visit_orders)
        token_type_embeds = self.token_type_embeddings(token_type_ids_batch)

        inputs_embeds = torch.cat(
            (inputs_embeds, time_stamps_embeds, ages_embeds),
            dim=-1,
        )

        inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        embeddings = inputs_embeds + token_type_embeds
        embeddings += visit_order_embeds
        embeddings += visit_segments_embeds

        embeddings = self.dropout(embeddings)

        return self.LayerNorm(embeddings)
