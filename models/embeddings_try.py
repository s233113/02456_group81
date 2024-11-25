import torch
from torch import nn
from transformers import MambaConfig
import pdb
from transformers.models.mamba.modeling_mamba import (
    MambaCausalLMOutput,
    MambaForCausalLM,
)
import numpy as np
from mortality_part_preprocessing import PairedDataset


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
# class ConceptEmbedding(nn.Module):
#     def __init__(self, num_features: int, embedding_size: int):
#         super().__init__()
#         self.embedding = nn.Embedding(num_features, embedding_size)

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         print("inputs shape in concept embedding;")
#         print(inputs.shape)

#         print(inputs)
#         print(torch.max(inputs))
#         print(torch.max(inputs.long()))
#         return self.embedding(inputs.long())

class ConceptEmbedding(nn.Module):
    def __init__(self, num_features, embedding_size, dropout_prob=0.5):
        super(ConceptEmbedding, self).__init__()
        
        # Define the linear layer
        self.fc = nn.Linear(num_features, embedding_size)  # num_features -> embedding_size
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        # Assuming input shape is [batch_size, num_features, time_steps]
        batch_size, num_features, time_steps = inputs.size()
        
        # Reshape inputs to [batch_size * time_steps, num_features] for linear layer
        inputs_reshaped = inputs.reshape(-1, num_features)  # Use reshape instead of view
        
        # Pass through the fully connected layer
        embeddings_reshaped = self.fc(inputs_reshaped)
        
        # Apply dropout
        embeddings_reshaped = self.dropout(embeddings_reshaped)
        
        # Reshape back to [batch_size, time_steps, embedding_size]
        embeddings = embeddings_reshaped.reshape(batch_size, time_steps, -1)

        return embeddings

# Full embedding layer
class MambaEmbeddingLayer(nn.Module):
    def __init__(self, config: MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").config, num_features: int, static_size: int, max_time_steps: int, dropout: float):
        super().__init__()
        self.num_features = num_features
        self.max_time_steps = max_time_steps
        self.embedding_size = config.hidden_size
        #self.embedding_size = 128


        self.time_embedding = TimeEmbeddingLayer(embedding_size=max_time_steps, is_time_delta=True) #debug
        #self.feature_embedding = ConceptEmbedding(num_features=128, embedding_size=max_time_steps)
        self.feature_embedding = ConceptEmbedding(num_features=num_features, embedding_size=self.embedding_size)
        self.static_embedding = StaticEmbedding(input_size=static_size, output_size=self.embedding_size)

        self.scale_layer = nn.Linear(
            self.embedding_size + max_time_steps,  # Combine feature and time embeddings
            self.embedding_size,
        )

        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, ts_values: torch.Tensor, ts_indicators: torch.Tensor, ts_times: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    #     # Time embeddings
    #     time_embeds = self.time_embedding(ts_times)

    #     # Feature embeddings
    #     ts_values_embedded = self.feature_embedding(ts_values)
    #     print("before ts_values_embedded: ")
    #     print(ts_values_embedded.shape)
    #     print(ts_indicators.shape)

    #      # Align ts_indicators for multiplication
    #     ts_indicators = ts_indicators.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, time_steps, num_features, 1]
        
    #     print("after unsqueze" , ts_indicators.shape)

    #     ts_values_embedded = ts_values_embedded * ts_indicators  # Element-wise multiplication (broadcasting)
    #     #ts_values_embedded = ts_values_embedded * ts_indicators.unsqueeze(-1)

    #     # Combine time and feature embeddings
    #     ts_combined = torch.cat((ts_values_embedded, time_embeds), dim=-1)
    #     ts_embeds = self.scale_layer(ts_combined)

    #     # Static embeddings
    #     static_embeds = self.static_embedding(static)

    #     # Add static embeddings and normalize
    #     combined_embeds = ts_embeds + static_embeds.unsqueeze(1)
    #     combined_embeds = self.LayerNorm(combined_embeds)
    #     combined_embeds = self.dropout(combined_embeds)

    #     return combined_embeds

    def forward(self, ts_values: torch.Tensor, ts_indicators: torch.Tensor, ts_times: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # Apply mask to ts_values before embedding
        ts_values_masked = ts_values * ts_indicators  # Element-wise masking [128, 37, 171]
        
        # Time embeddings
        time_embeds = self.time_embedding(ts_times)  # [128, 171, time_embedding_dim]

        # Feature embeddings
        ts_values_embedded = self.feature_embedding(ts_values_masked)  # [128, 171, 768]

        # Combine time and feature embeddings
        ts_combined = torch.cat((ts_values_embedded, time_embeds), dim=-1)  # [128, 171, 768 + time_embedding_dim]
        ts_embeds = self.scale_layer(ts_combined)

        # Static embeddings
        static_embeds = self.static_embedding(static)  # [128, hidden_size]

        # Add static embeddings and normalize
        combined_embeds = ts_embeds + static_embeds.unsqueeze(1)  # Broadcast static to time dimension
        combined_embeds = self.LayerNorm(combined_embeds)
        combined_embeds = self.dropout(combined_embeds)

        return combined_embeds

# # Preprocessing and embedding function
# def preprocess_and_embed(train_data, config, dropout):
#     """
#     Preprocess the train_data and embed it for model input.

#     Parameters:
#         train_data: numpy array of dictionaries
#         config: MambaConfig object with model hyperparameters

#     Returns:
#         Embedded data and labels
#     """
#     # pdb.set_trace()
#     max_time_steps = max(len(entry["ts_times"]) for entry in train_data)
#     num_features = train_data[0]["ts_values"].shape[1]
#     static_size = len(train_data[0]["static"])

#     embedding_layer = MambaEmbeddingLayer(config, num_features, static_size, max_time_steps,dropout)

#     # # Preprocess data
#     # ts_values = [torch.tensor(entry["ts_values"], dtype=torch.float32) for entry in train_data]
#     # ts_indicators = [torch.tensor(entry["ts_indicators"], dtype=torch.float32) for entry in train_data]
#     # ts_times = [torch.tensor(entry["ts_times"], dtype=torch.float32) for entry in train_data]
#     # static_features = [torch.tensor(entry["static"], dtype=torch.float32) for entry in train_data]
#     # labels = [torch.tensor(entry["labels"], dtype=torch.long) for entry in train_data]
#     # # static_features = torch.tensor([entry["static"] for entry in train_data], dtype=torch.float32)
#     # # labels = torch.tensor([entry["labels"] for entry in train_data], dtype=torch.long)

#     # # Pad sequences to the same length
#     # ts_values_padded = nn.utils.rnn.pad_sequence(ts_values, batch_first=True)
#     # ts_indicators_padded = nn.utils.rnn.pad_sequence(ts_indicators, batch_first=True)
#     # ts_times_padded = nn.utils.rnn.pad_sequence(ts_times, batch_first=True)

#     # Create embeddings
#     embedded_data = embedding_layer(ts_values_padded, ts_indicators_padded, ts_times_padded, static_features)

#     return embedded_data, labels

def preprocess_and_embed(preprocessed_data, train_data_loader, config, dropout):
    """
    Embed preprocessed data for model input.

    Parameters:
        preprocessed_data: A dataset object (e.g., train_pair, val_data, test_data) containing preprocessed data.
        config: MambaConfig object with model hyperparameters.
        dropout: Dropout rate.

    Returns:
        Embedded data and labels
    """

    # max_time_steps_pos = np.shape(preprocessed_data.dataset_pos.times_array)[1] # Number of time steps
    # num_features_pos = np.shape(preprocessed_data.dataset_pos.data_array)[1]  # Number of features
    # static_size_pos = np.shape(preprocessed_data.dataset_pos.static_array)[1]  # Number of static features
   
    # max_time_steps_neg = np.shape(preprocessed_data.dataset_neg.times_array)[1] # Number of time steps
    # num_features_neg = np.shape(preprocessed_data.dataset_neg.data_array)[1]  # Number of features
    # static_size_neg = np.shape(preprocessed_data.dataset_neg.static_array)[1]  # Number of static features
    
    # max_time_steps=np.maximum(max_time_steps_pos, max_time_steps_neg)
    # num_features=np.maximum(num_features_neg, num_features_pos)
    # static_size=np.maximum(static_size_neg, static_size_pos)

    # print(max_time_steps)
    # print(num_features)
    # print(static_size)

    data, times, static, labels, mask, delta=next(iter(train_data_loader))

    ts_values = data  # (Batch, Time, Features)
    ts_indicators = mask  # (Batch, Time, Features)
    ts_times = times  # (Batch, Time)
    static_features = static  # (Batch, Static Features)
    labels = labels  # (Batch,)

    print("ts_values:")
    print("type: ", type(ts_values))
    print("shape: ", ts_values.shape)
    print(ts_values)

    print("times shape:")
    print(times.shape)
    print("data shape:")
    print(data.shape)
    print("static shape:")
    print(static.shape)
    max_time_steps=times.shape[1]
    num_features=data.shape[1]
    #num_features = int(ts_values.max().item() +1)

    
    static_size=static.shape[1]

    print("max time steps, num features and static size:")
    print(max_time_steps)
    print(num_features)
    print(static_size)

    print("hidden size config:")
    print(config.hidden_size)

    # Initialize the embedding layer
    
    embedding_layer = MambaEmbeddingLayer(config, num_features, static_size, max_time_steps, dropout)

    # test=PairedDataset.paired_collate_fn(batch(preprocessed_data.dataset_neg, preprocessed_data.dataset_pos))
    # Extract preprocessed data tensors
    # ts_values = preprocessed_data.dataset_neg.data_array  # (Batch, Time, Features)
    # ts_indicators = preprocessed_data.dataset_neg.sensor_mask_array  # (Batch, Time, Features)
    # ts_times = preprocessed_data.dataset_neg.times_array  # (Batch, Time)
    # static_features = preprocessed_data.dataset_neg.static_array  # (Batch, Static Features)
    # labels = preprocessed_data.dataset_neg.label_array  # (Batch,)

    #data, times, static, labels, mask, delta=next(iter(train_data_loader))
    
    print("dimensions before entering embedding layer")
    print(data.shape)
    print(mask.shape)
    print(times.shape)
    print(static.shape)
    print(labels.shape)
    print("entering embedding layer line 219")

    print("Maximum index in ts_values:", ts_values.max().item())
    print("Minimum index in ts_values:", ts_values.min().item())
    print("Embedding vocabulary size (num_features):", num_features)

    # Create embeddings

    #ERROR HERE
    embedded_data = embedding_layer(ts_values, ts_indicators, ts_times, static_features)

    return embedded_data,labels

def get_vocab_size(embedding_layer: ConceptEmbedding) -> int:
    """
    Return the number of possible tokens in the vocabulary.
 
    Parameters:
        embedding_layer (ConceptEmbedding): The concept embedding layer.
 
    Returns:
        int: Number of tokens in the vocabulary.
    """
    return embedding_layer.embedding.num_embeddings
# train_data = np.load('../split_1/train_physionet2012_1.npy', allow_pickle=True)
# print(preprocess_and_embed(train_data, MambaConfig))