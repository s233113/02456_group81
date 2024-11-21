import torch
import torch.nn as nn
import torch.jit as jit


def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """

    if mask is not None:
        # eliminate all values learned from nonexistant timepoints
        mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
        data_summed = torch.sum(datatensor * mask_expanded, dim=1)

        # find out number of existing timepoints
        data_counts = mask_expanded.sum(1)
        data_counts = torch.clamp(data_counts, min=1e-9)  # put on min clamp

        # Calculate average:
        averaged = data_summed / (data_counts)
    else:
        averaged = datatensor.mean(dim=1)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """

    if mask is not None:

        # eliminate all values learned from nonexistant timepoints
        mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

        datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    maxed = torch.max(datatensor, 1)[0]

    return maxed


class SingleChannelInterp(jit.ScriptModule):
    def __init__(self, input_dim):
        super(SingleChannelInterp, self).__init__()
        assert input_dim % 3 == 0
        self.d_dim = input_dim // 3
        self.kernel = nn.Parameter(torch.zeros(self.d_dim))

    @jit.script_method
    def forward(self, x, interpolation_grid, reconstruction: bool = False):
        x_t = x[:, : self.d_dim, :]
        m = x[:, self.d_dim : 2 * self.d_dim, :]
        d = x[:, 2 * self.d_dim : 3 * self.d_dim, :]
        time_stamp = x.shape[2]

        if reconstruction:
            ref_t = d.unsqueeze(2).expand(-1, -1, time_stamp, -1)
            output_dim = time_stamp
        else:
            ref_t = interpolation_grid.unsqueeze(1).unsqueeze(1)  # Expand grid
            output_dim = interpolation_grid.shape[-1]

        d = d.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        mask = m.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        x_t = x_t.unsqueeze(-1).expand(-1, -1, -1, output_dim)

        norm = (d - ref_t) ** 2
        a = torch.ones((self.d_dim, time_stamp, output_dim), device=x.device)
        pos_kernel = torch.log(1 + torch.exp(self.kernel))  # Positive kernel
        alpha = a * pos_kernel.view(-1, 1, 1)

        w = torch.logsumexp(-alpha * norm + torch.log(mask + 1e-9), dim=2)
        w1 = w.unsqueeze(2).expand(-1, -1, time_stamp, -1)
        w1 = torch.exp(-alpha * norm + torch.log(mask + 1e-9) - w1)

        y = (w1 * x_t).sum(dim=2)

        if reconstruction:
            rep1 = torch.cat([y, w], dim=1)
        else:
            w_t = torch.logsumexp(-10.0 * alpha * norm + torch.log(mask + 1e-9), dim=2)
            w_t = w_t.unsqueeze(2).expand(-1, -1, time_stamp, -1)
            w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask + 1e-9) - w_t)
            y_trans = (w_t * x_t).sum(dim=2)
            rep1 = torch.cat([y, w, y_trans], dim=1)

        return rep1


class CrossChannelInterp(jit.ScriptModule):
    def __init__(self, input_dim):
        super(CrossChannelInterp, self).__init__()
        self.d_dim = input_dim // 3
        self.cross_channel_interp = nn.Parameter(torch.eye(self.d_dim))

    @jit.script_method
    def forward(self, x, reconstruction: bool = False):
        output_dim = x.shape[-1]
        y = x[:, : self.d_dim, :]
        w = x[:, self.d_dim : 2 * self.d_dim, :]
        intensity = torch.exp(w)

        y = y.transpose(1, 2)
        w = w.transpose(1, 2)
        w2 = w

        w = w.unsqueeze(-1).expand(-1, -1, -1, self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)

        mean = y.mean(dim=1, keepdim=True)
        mean = mean.expand(-1, output_dim, -1)
        w2 = torch.matmul(w * (y - mean), self.cross_channel_interp) + mean

        rep1 = w2.transpose(1, 2)

        if not reconstruction:
            y_trans = x[:, 2 * self.d_dim : 3 * self.d_dim, :]
            y_trans = y_trans - rep1
            rep1 = torch.cat([rep1, intensity, y_trans], dim=1)

        return rep1


class InterpolationPredictionModel(nn.Module):
    def __init__(
        self,
        output_dims,
        recurrent_n_units,
        ipnets_imputation_stepsize,
        dropout,
        recurrent_dropout,
        ipnets_reconst_fraction,
        sensor_count,
        pooling="hidden", # One of hidden, mean, max
        **kwargs
    ):
        super(InterpolationPredictionModel, self).__init__()

        self.pool = pooling

        self.imputation_stepsize = ipnets_imputation_stepsize
        self.reconst_fraction = ipnets_reconst_fraction
        self.eps = 1e-9
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.n_units = recurrent_n_units

        self.interp_dim = sensor_count * 3

        self.single_channel_interp = SingleChannelInterp(input_dim=self.interp_dim)
        self.cross_channel_interp = CrossChannelInterp(input_dim=self.interp_dim)

        self.demo_encoder = nn.Sequential(
            nn.LazyLinear(recurrent_n_units), nn.ReLU(), nn.Linear(recurrent_n_units, recurrent_n_units)
        )

        self.gru = nn.GRU(input_size=self.interp_dim, hidden_size=recurrent_n_units, batch_first=True)

        self.output_layer = nn.Linear(recurrent_n_units, output_dims)

        self.input_dropout_layer = nn.Dropout(self.dropout)
        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout)

    def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:

        times, values, measurements, grid, grid_lengths, static = (
            self.create_timepoint_grid(x, static, time, sensor_mask)
        )

        layer_input = torch.cat((values, measurements.float(), times), dim=1)

        sic_output = self.single_channel_interp(layer_input, grid)
        crc_output = self.cross_channel_interp(sic_output)

        rnn_input = crc_output.transpose(1, 2)

        if self.training:
            dropout_mask = self.input_dropout_layer(torch.ones_like(rnn_input[:, 0, :]))
            rnn_input = rnn_input * dropout_mask.unsqueeze(1)

        demo_encoded = self.demo_encoder(static)
        hidden_state = demo_encoded

        rnn_output = self.gru(rnn_input, hidden_state.unsqueeze(0))[0]
        if self.training:
            dropout_mask = self.recurrent_dropout_layer(torch.ones_like(rnn_output))
            rnn_output = rnn_output * dropout_mask

        # Since we did not mask during the RNN pass, we take the last non-zero
        # time reading
        if self.pool == "hidden":
            idx = (
                (grid_lengths - 1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, 1, self.n_units)
                    .long()
            )
            pooled = rnn_output.gather(1, idx).squeeze(1)
        elif self.pool == "max":
            pooled = masked_max_pooling(rnn_output, None)
        elif self.pool == "mean":
            pooled = masked_mean_pooling(rnn_output, None)
        else:
            raise NotImplementedError(f"Pooling function {self.pool} not supported.")

        # Reconstruction loss calculation
        reconst_mask = (
            torch.rand(measurements.shape, device=measurements.device)
            > self.reconst_fraction
        )
        context_measurements = measurements & reconst_mask

        # Check for missing observations, fill first time step
        nothing_observed = context_measurements.sum(dim=-1, keepdim=True) == 0
        context_measurements = context_measurements | nothing_observed

        # Reconstruction pass; improve imputation
        reconst_input = torch.cat((values, context_measurements.float(), times), dim=1)
        sic_reconst = self.single_channel_interp(
            reconst_input, grid, reconstruction=True
        )
        crc_reconst = self.cross_channel_interp(sic_reconst, reconstruction=True)

        target_measurements = measurements & (~context_measurements)
        squared_error = target_measurements.float() * (values - crc_reconst) ** 2
        instance_wise_reconst_error = squared_error.sum(dim=[1, 2]) / (
            target_measurements.float().sum(dim=[1, 2]) + self.eps
        )

        # Loss calculation based on training phase
        if self.training:
            reconstruction_loss = instance_wise_reconst_error.mean()
        else:
            reconstruction_loss = torch.zeros(
                (), dtype=torch.float32, device=x[0].device
            )

        output = self.output_layer(pooled)

        return output, reconstruction_loss

    def create_timepoint_grid(self, x, static, time, sensor_mask):

        all_demo = static
        all_x = []
        all_y = []
        all_measurements = []
        all_grid = []
        all_grid_length = []

        end_time = torch.max(time)

        for batch_ind in range(x.shape[0]):
            # Bit of notation wonkiness; x is actually considered the time with y the value.
            y_ind = x[batch_ind].permute(1, 0)
            x_ind = time[batch_ind]
            sensor_mask_ind = sensor_mask[batch_ind].permute(1, 0).bool()

            length = torch.nonzero(sensor_mask_ind.sum(dim=0))

            X = x_ind.unsqueeze(-1)

            # Check if a value was never measured. If this is the case, add an
            # observation at timepoint t=0 with the mean, assuming mean-centered data (mean = 0).
            n_observed_values = (sensor_mask_ind == False).sum(dim=0)
            nothing_ever_observed = torch.where(n_observed_values == length)[0]

            # Update Y and measurements to add observations at t=0
            indices = torch.stack(
                [torch.zeros_like(nothing_ever_observed), nothing_ever_observed], dim=1
            )
            Y = y_ind.index_put(
                (indices[:, 0], indices[:, 1]),
                torch.zeros(indices.shape[0], device=y_ind.device),
            )
            measurements = sensor_mask_ind.index_put(
                (indices[:, 0], indices[:, 1]),
                torch.ones(
                    indices.shape[0], dtype=torch.bool, device=sensor_mask_ind.device
                ),
            )

            # Generate a grid for imputation
            # End time + self.imputation_stepsize would, in the reference implementation be the
            # "hours_look_ahead" parameter https://github.com/mlds-lab/interp-net/blob/master/src/interpolation_layer.py
            # Since we want to infer values for all timestamps in our data, we set the lookahead to be however
            # many hours are in our data.
            # Similarly, since torch.arange and np.linspace work slightly differently, we have
            # self.ref_points == (end_time + self.imputation_stepsize) // self.imputation_stepsize
            # Which means that in our case, self.ref_points = (48 hours // self.imputation_stepsize) + 1
            # Self.imputation_stepsize = 0.25 then means we have 193 ref points.
            grid = torch.arange(
                0,
                end_time + self.imputation_stepsize,
                step=self.imputation_stepsize,
                device=x_ind.device,
            )
            grid_length = torch.tensor(
                grid.shape[0], dtype=torch.int32, device=grid.device
            )

            X = X.repeat(1, Y.shape[-1])

            X = X.transpose(0, 1)
            Y = Y.transpose(0, 1)
            measurements = measurements.transpose(0, 1)

            all_y.append(Y)
            all_x.append(X)
            all_measurements.append(measurements)
            all_grid.append(grid)
            all_grid_length.append(grid_length)

        all_grid = torch.stack(all_grid)
        all_y = torch.stack(all_y)
        all_x = torch.stack(all_x)
        all_measurements = torch.stack(all_measurements)
        all_grid_length = torch.stack(all_grid_length)

        return all_x, all_y, all_measurements, all_grid, all_grid_length, all_demo
