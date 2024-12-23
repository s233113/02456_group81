"""Utilities following HuggingFace style for Mamba models."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.mamba.modeling_mamba import (
    MAMBA_INPUTS_DOCSTRING,
    MAMBA_START_DOCSTRING,
    MambaPreTrainedModel
)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

_CONFIG_FOR_DOC = "MambaConfig"

@dataclass
class MambaSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of Mamba sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# ==================================== SingleHead ================================================= #


class MambaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        """Initialize the head."""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, features, **kwargs):
        """Forward pass."""
        x = features  # Pooling is done by the forward pass

        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)

        return self.out_proj(x)


@add_start_docstrings(
    """
    Mamba Model with a sequence classification/regression head on top
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.
    """,
    MAMBA_START_DOCSTRING,
)
class MambaForSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config, pretrained_model):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pretrained_model = pretrained_model
        self.backbone = self.pretrained_model.backbone
        self.classifier = MambaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        MAMBA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=MambaSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MambaSequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
        """

        if inputs_embeds is not None:
            sequence_outputs = self.backbone(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            sequence_outputs = self.backbone(
                input_ids=input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


        last_hidden_states = sequence_outputs[0]
        batch_size = last_hidden_states.shape[0]

        # Pool the hidden states for the last tokens before padding
        # to use for classification
        last_token_indexes = (
            torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        )
        pooled_last_hidden_states = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            last_token_indexes,
        ]

        logits = self.classifier(pooled_last_hidden_states)

        loss= None
        
        return loss,logits

