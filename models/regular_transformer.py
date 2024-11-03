import torch
import numpy as np
from torch import nn
from x_transformers import Encoder


def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
    data_summed = torch.sum(datatensor * mask_expanded, dim=1)

    # find out number of existing timepoints
    data_counts = mask_expanded.sum(1)
    data_counts = torch.clamp(data_counts, min=1e-9)  # put on min clamp

    # Calculate average:
    averaged = data_summed / (data_counts)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

    datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    maxed = torch.max(datatensor, 1)[0]

    return maxed


class PositionalEncodingTF(nn.Module):
    """
    Based on the SEFT positional encoding implementation
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2

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
        return pe


class EncoderClassifierRegular(nn.Module):

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        use_static=True,
        obs_strategy="both",
        **kwargs
    ):
        super().__init__()

        self.obs_strategy = obs_strategy  # BINARY
        self.pooling = pooling
        self.device = device
        self.sensors_count = sensors_count
        # self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count

        if self.obs_strategy in ("indicator_only", "obs_only"):  # BINARY
            # print("binary_values_only")
            self.sensor_axis_dim_in = self.sensors_count
        else:  # BINARY
            self.sensor_axis_dim_in = 2 * self.sensors_count

        self.sensor_axis_dim = self.sensor_axis_dim_in
        if self.sensor_axis_dim % 2 != 0:
            self.sensor_axis_dim += 1

        # self.time_axis_dim_in = 2 * self.max_timepoint_count
        # self.time_axis_dim = min(2 * self.max_timepoint_count, 500)
        self.static_out = self.static_count + 4

        self.attn_layers_2 = Encoder(
            dim=self.sensor_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
        )

        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)

        self.use_static = use_static

        if self.use_static:
            self.static_embedding = nn.Linear(self.static_count, self.static_out)
            self.nonlinear_merger = nn.Linear(
                self.sensor_axis_dim + self.static_out,
                self.sensor_axis_dim + self.static_out,
            )
            self.classifier = nn.Linear(
                self.sensor_axis_dim + self.static_out, num_classes
            )
        else:
            self.nonlinear_merger = nn.Linear(
                self.sensor_axis_dim,
                self.sensor_axis_dim,
            )
            self.classifier = nn.Linear(self.sensor_axis_dim, num_classes)

        self.pos_encoder = PositionalEncodingTF(self.sensor_axis_dim)

    def forward(self, x, static, time, sensor_mask, **kwargs):

        x_time = torch.clone(x)  # (N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1))  # (N, T, F)
        mask = (
            torch.count_nonzero(x_time, dim=2)
        ) > 0  # mask for sum of all sensors for each person/at each timepoint

        # add indication for missing sensor values
        x_sensor_mask = torch.clone(sensor_mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)
        if self.obs_strategy == "indicator_only":  # Binary
            x_time = x_sensor_mask.float()  # Binary
        elif self.obs_strategy == "obs_only":
            x_time = x_time
        elif self.obs_strategy == "both":  # Binary
            x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F) #Binary
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")
        # make sensor embeddings
        x_time = self.sensor_embedding(x_time)  # (N, T, F)
        # add positional encodings
        pe = self.pos_encoder(time).to(self.device)  # taken from RAINDROP, (N, T, pe)
        x_time = torch.add(x_time, pe)  # (N, T, F) (N, F)
        # run time attention
        x_time = self.attn_layers_2(x_time, mask=mask)

        if self.pooling == "mean":
            x_time = masked_mean_pooling(x_time, mask)
        elif self.pooling == "median":
            x_time = torch.median(x_time, dim=1)[0]
        elif self.pooling == "sum":
            x_time = torch.sum(x_time, dim=1)  # sum on time
        elif self.pooling == "max":
            x_time = masked_max_pooling(x_time, mask)

        # concatenate poolingated attented tensors
        if self.use_static:
            static = self.static_embedding(static)
            x_merged = torch.cat((x_time, static), axis=1)
        else:
            x_merged = x_time

        nonlinear_merged = self.nonlinear_merger(x_merged).relu()

        # classify!
        return self.classifier(nonlinear_merged)
