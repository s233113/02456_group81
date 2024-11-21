import torch
from torch import nn
from transformers import MambaConfig


# Time embedding layer
class TimeEmbeddingLayer(nn.Module):
    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> torch.Tensor:
        if self.is_time_delta:
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi
        return torch.sin(next_input)


# Static embedding layer
class StaticEmbedding(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, static_inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(static_inputs)


# Feature embedding layer
class ConceptEmbedding(nn.Module):
    def __init__(self, num_features: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(inputs.long())


# Full embedding layer
class MambaEmbeddingLayer(nn.Module):
    def __init__(self, config: MambaConfig, num_features: int, static_size: int, max_time_steps: int):
        super().__init__()
        self.num_features = num_features
        self.max_time_steps = max_time_steps
        self.embedding_size = config.hidden_size

        self.time_embedding = TimeEmbeddingLayer(embedding_size=32, is_time_delta=True)
        self.feature_embedding = ConceptEmbedding(num_features=num_features, embedding_size=config.hidden_size)
        self.static_embedding = StaticEmbedding(input_size=static_size, output_size=config.hidden_size)

        self.scale_layer = nn.Linear(
            config.hidden_size + 32,  # Combine feature and time embeddings
            config.hidden_size,
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ts_values: torch.Tensor, ts_indicators: torch.Tensor, ts_times: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # Time embeddings
        time_embeds = self.time_embedding(ts_times)

        # Feature embeddings
        ts_values_embedded = self.feature_embedding(ts_values)
        ts_values_embedded = ts_values_embedded * ts_indicators.unsqueeze(-1)

        # Combine time and feature embeddings
        ts_combined = torch.cat((ts_values_embedded, time_embeds), dim=-1)
        ts_embeds = self.scale_layer(ts_combined)

        # Static embeddings
        static_embeds = self.static_embedding(static)

        # Add static embeddings and normalize
        combined_embeds = ts_embeds + static_embeds.unsqueeze(1)
        combined_embeds = self.LayerNorm(combined_embeds)
        combined_embeds = self.dropout(combined_embeds)

        return combined_embeds


# Preprocessing and embedding function
def preprocess_and_embed(train_data, config):
    """
    Preprocess the train_data and embed it for model input.

    Parameters:
        train_data: numpy array of dictionaries
        config: MambaConfig object with model hyperparameters

    Returns:
        Embedded data and labels
    """
    max_time_steps = max(len(entry["ts_times"]) for entry in train_data)
    num_features = train_data[0]["ts_values"].shape[1]
    static_size = len(train_data[0]["static"])

    embedding_layer = MambaEmbeddingLayer(config, num_features, static_size, max_time_steps)

    # Preprocess data
    ts_values = [torch.tensor(entry["ts_values"], dtype=torch.float32) for entry in train_data]
    ts_indicators = [torch.tensor(entry["ts_indicators"], dtype=torch.float32) for entry in train_data]
    ts_times = [torch.tensor(entry["ts_times"], dtype=torch.float32) for entry in train_data]
    static_features = torch.tensor([entry["static"] for entry in train_data], dtype=torch.float32)
    labels = torch.tensor([entry["labels"] for entry in train_data], dtype=torch.long)

    # Pad sequences to the same length
    ts_values_padded = nn.utils.rnn.pad_sequence(ts_values, batch_first=True)
    ts_indicators_padded = nn.utils.rnn.pad_sequence(ts_indicators, batch_first=True)
    ts_times_padded = nn.utils.rnn.pad_sequence(ts_times, batch_first=True)

    # Create embeddings
    embedded_data = embedding_layer(ts_values_padded, ts_indicators_padded, ts_times_padded, static_features)

    return embedded_data, labels
