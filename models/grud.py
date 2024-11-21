import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from typing import List


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


def exp_relu(x):
    return torch.exp(-F.relu(x))


def get_activation(identifier):
    if identifier is None:
        return None
    if identifier == "exp_relu":
        return exp_relu
    return getattr(F, identifier.lower(), None)


@jit.script
def generate_masks(inputs: torch.Tensor, dropout_prob: float, training: bool, num_units: int, count: int):
    masks = torch.jit.annotate(List[torch.Tensor], [])  # Predefine the list type for TorchScript

    for _ in range(count):
        if not training or dropout_prob == 0:
            mask = torch.ones_like(inputs[:, :num_units], dtype=torch.float32)
        else:
            # Apply dropout
            mask = F.dropout(
                torch.ones_like(inputs[:, :num_units], dtype=torch.float32),
                p=dropout_prob,
                training=training,
            )
        masks.append(mask)

    return torch.stack(masks)  # Stack the masks into a tensor at the end


class GRUDCell(jit.ScriptModule):
    # NOTE: These masking functions are necessary because they keep the mask persistent
    # throughout the entire time-series for a single data point. If we were to use F.dropout as
    # a replacement, each entry in the time-series would receive a different dropout mask, which
    # is undesirable.
    def get_dropout_mask_for_cell(
        self, inputs, dropout_prob: float, training: bool, num_units: int, count: int
    ):
        if not self.input_dropout_masks[0].numel() == 1:
            return self.input_dropout_masks
        else:
            return generate_masks(inputs, dropout_prob, training, num_units, count)

    def get_rdropout_mask_for_cell(
        self, inputs, dropout_prob: float, training: bool, num_units: int, count: int
    ):
        if self.recurrent_dropout_masks[0].numel() == 1:
            return self.recurrent_dropout_masks
        else:
            return generate_masks(inputs, dropout_prob, training, num_units, count)

    def get_mdropout_mask_for_cell(
        self, inputs, dropout_prob: float, training: bool, num_units: int, count: int
    ):
        if self.feed_dropout_masks[0].numel() == 1:
            return self.feed_dropout_masks
        else:
            return generate_masks(inputs, dropout_prob, training, num_units, count)

    def reset_masks(self):
        self.recurrent_dropout_masks = torch.tensor([1, 1, 1])
        self.input_dropout_masks = torch.tensor([1, 1, 1])
        self.feed_dropout_masks = torch.tensor([1, 1, 1])

    def __init__(
        self,
        input_size,
        hidden_size,
        device,
        x_imputation="zero",
        input_decay="exp_relu",
        hidden_decay="exp_relu",
        activation="tanh",
        recurrent_activation="hardsigmoid",
        use_decay_bias=True,
        feed_masking=True,
        masking_decay=None,
        dropout=0.0,
        recurrent_dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.x_imputation = x_imputation
        self.input_decay = get_activation(input_decay)
        self.hidden_decay = get_activation(hidden_decay)
        self.activation = get_activation(activation)
        self.recurrent_activation = get_activation(recurrent_activation)
        self.use_decay_bias = use_decay_bias
        self.feed_masking = feed_masking
        self.masking_decay = get_activation(masking_decay)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.use_input_decay: bool = bool(input_decay)
        self.use_hidden_decay: bool = bool(hidden_decay)
        self.use_masking_decay: bool = bool(masking_decay)

        self.input_dropout_masks = torch.tensor([1, 1, 1])
        self.recurrent_dropout_masks = torch.tensor([1, 1, 1])
        self.feed_dropout_masks = torch.tensor([1, 1, 1])

        self.kernel_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.kernel_r = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.kernel_h = nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.recurrent_kernel_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.recurrent_kernel_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.recurrent_kernel_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))

        if self.use_input_decay:
            self.input_decay_kernel = nn.Parameter(torch.Tensor(input_size))
            if self.use_decay_bias:
                self.input_decay_bias = nn.Parameter(torch.Tensor(input_size))

        if self.use_hidden_decay:
            self.hidden_decay_kernel = nn.Parameter(
                torch.Tensor(input_size, hidden_size)
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = nn.Parameter(torch.Tensor(hidden_size))

        if self.feed_masking:
            self.masking_kernel_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.masking_kernel_r = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.masking_kernel_h = nn.Parameter(torch.Tensor(hidden_size, input_size))

            if self.use_masking_decay:
                self.masking_decay_kernel = nn.Parameter(torch.Tensor(input_size))
                if self.use_decay_bias:
                    self.masking_decay_bias = nn.Parameter(torch.Tensor(input_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel_z)
        nn.init.xavier_uniform_(self.kernel_h)
        nn.init.xavier_uniform_(self.kernel_r)
        nn.init.xavier_uniform_(self.recurrent_kernel_z)
        nn.init.xavier_uniform_(self.recurrent_kernel_h)
        nn.init.xavier_uniform_(self.recurrent_kernel_r)

        if self.use_input_decay is not None:
            nn.init.zeros_(self.input_decay_kernel)
            if self.use_decay_bias:
                nn.init.zeros_(self.input_decay_bias)

        if self.use_hidden_decay is not None:
            nn.init.zeros_(self.hidden_decay_kernel)
            if self.use_decay_bias:
                nn.init.zeros_(self.hidden_decay_bias)

        if self.feed_masking:
            nn.init.xavier_uniform_(self.masking_kernel_z)
            nn.init.xavier_uniform_(self.masking_kernel_h)
            nn.init.xavier_uniform_(self.masking_kernel_r)
            if self.masking_decay is not None:
                nn.init.zeros_(self.masking_decay_kernel)
                if self.use_decay_bias:
                    nn.init.zeros_(self.masking_decay_bias)

    @jit.script_method
    def forward(self, input_x, input_m, input_s, h_tm1, x_keep_tm1, s_prev_tm1):

        input_1m = 1.0 - input_m.float()
        input_d = input_s - s_prev_tm1

        self.input_dropout_masks = self.get_dropout_mask_for_cell(
            input_x,
            dropout_prob=self.dropout,
            training=self.training,
            num_units=self.input_size,
            count=3,
        )
        self.recurrent_dropout_masks = self.get_rdropout_mask_for_cell(
            h_tm1,
            dropout_prob=self.recurrent_dropout,
            training=self.training,
            num_units=self.hidden_size,
            count=3,
        )
        self.feed_dropout_masks = self.get_mdropout_mask_for_cell(
            input_m.double(),
            dropout_prob=self.dropout,
            training=self.training,
            num_units=self.input_size,
            count=3,
        )

        gamma_di = torch.tensor(1)
        gamma_dh = torch.tensor(1)
        gamma_dm = torch.tensor(1)

        if self.use_input_decay:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = gamma_di + self.input_decay_bias
            gamma_di = self.input_decay(gamma_di)

        if self.use_hidden_decay:
            gamma_dh = torch.matmul(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = gamma_dh + self.hidden_decay_bias
            gamma_dh = self.hidden_decay(gamma_dh)

        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = gamma_dm + self.masking_decay_bias
            gamma_dm = self.masking_decay(gamma_dm)

        # fix for device error
        input_m = input_m.to(self.device)
        input_x = input_x.to(self.device)
        x_keep_tm1 = x_keep_tm1.to(self.device)
        gamma_di = gamma_di.to(self.device)

        if self.use_input_decay:
            x_keep_t = torch.where(input_m, input_x, x_keep_tm1)
            x_t = torch.where(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == "forward":
            x_t = torch.where(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == "zero":
            x_t = torch.where(input_m, input_x, torch.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == "raw":
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError(f"Invalid x_imputation: {self.x_imputation}")

        if self.use_hidden_decay:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        m_t = torch.tensor(1)
        m_z = torch.tensor(1)
        m_h = torch.tensor(1)
        m_r = torch.tensor(1)

        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        if self.training:
            x_z, x_r, x_h = (
                x_t * self.input_dropout_masks[0],
                x_t * self.input_dropout_masks[1],
                x_t * self.input_dropout_masks[2],
            )
            h_tm1_z, h_tm1_r = (
                h_tm1d * self.recurrent_dropout_masks[0],
                h_tm1d * self.recurrent_dropout_masks[1],
            )
            if self.feed_masking:
                m_z, m_r, m_h = (
                    m_t * self.feed_dropout_masks[0],
                    m_t * self.feed_dropout_masks[1],
                    m_t * self.feed_dropout_masks[2],
                )
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t

        z_t = F.linear(x_z, self.kernel_z) + F.linear(h_tm1_z, self.recurrent_kernel_z)
        r_t = F.linear(x_r, self.kernel_r) + F.linear(h_tm1_r, self.recurrent_kernel_r)
        hh_t = F.linear(x_h, self.kernel_h)
        if self.feed_masking:
            z_t += F.linear(m_z, self.masking_kernel_z)
            r_t += F.linear(m_r, self.masking_kernel_r)
            hh_t += F.linear(m_h, self.masking_kernel_h)
        else:
            z_t = z_t + self.bias_z
            r_t = r_t + self.bias_r
            hh_t = hh_t + self.bias_h
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)

        h_hm1_t = r_t * h_tm1d * self.recurrent_dropout_masks[2]
        hh_t = self.activation(hh_t + F.linear(h_hm1_t, self.recurrent_kernel_h))

        h_t = z_t * h_tm1 + (1 - z_t) * hh_t

        s_prev_t = torch.where(input_m, input_s.expand(-1, self.input_size), s_prev_tm1)

        return h_t, x_keep_t, s_prev_t


class GRUD(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, device, **kwargs):
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size, device, **kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

    @jit.script_method
    def forward(self, values, mask, time, h_t=None, x_keep_t=None, s_prev_t=None):
        batch_size, seq_len, _ = values.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=values.device)
            x_keep_t = torch.zeros(batch_size, self.input_size, device=values.device)
            s_prev_t = time[:, 0].expand(-1, self.input_size)

        outputs = []

        # Sucks to have to do this, but at least we can crank the batch size way up
        # Would be best to find a more efficient RNN module that does allow for a
        # custom cell
        for t in range(seq_len):
            x_t = values[:, t].to(self.device)
            m_t = mask[:, t].to(self.device)
            s_t = time[:, t].to(self.device)

            h_t, x_keep_t, s_prev_t = self.cell(
                x_t, m_t, s_t, h_t, x_keep_t, s_prev_t
            )
            outputs.append(h_t.cpu())
        self.cell.reset_masks()

        return torch.stack(outputs, dim=1)


class GRUDModel(nn.Module):
    def __init__(
        self,
        input_dim,
        static_dim,
        output_dims,
        recurrent_n_units,
        dropout,
        recurrent_dropout,
        device,
        pooling="hidden", # One of hidden, mean, max
        **kwargs,
    ):
        super().__init__()
        self.device = device
        # Set up model parameters
        self.n_units = recurrent_n_units
        self.input_dim = input_dim
        self.pool = pooling

        # Define layers
        self.rnn = GRUD(
            input_dim,
            recurrent_n_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            device=self.device,
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, recurrent_n_units),
            nn.ReLU(),
            nn.Linear(recurrent_n_units, recurrent_n_units),
        )

        self.output_layer = nn.Linear(recurrent_n_units, output_dims)

    def forward(self, x, static, time, sensor_mask, **kwargs):

        values = torch.permute(x, (0, 2, 1))
        sensor_mask = torch.permute(sensor_mask, (0, 2, 1)).bool()

        static = static.to(self.device)
        static_encoded = self.static_encoder(static)

        # Prepare time
        time = time.unsqueeze(-1)

        # Initial state
        h_t = static_encoded
        x_keep_t = torch.zeros(static.size(0), self.input_dim, device=static.device)
        s_prev_t = time[:, 0].repeat(1, self.input_dim).to(self.device)

        # Creates padding mask. In RNN, those are used to ensure we do not update the hidden state
        # for #masked out timesteps. In practice, we can just ensure that we grab the hidden state
        # from the last non-masked timepoint instead.
        time_mask = (torch.count_nonzero(time, dim=2)) > 0
        time_mask[:, 0] = True  #0th timepoint is always valid

        grud_output = self.rnn(values, sensor_mask, time, h_t, x_keep_t, s_prev_t)

        if self.pool == "hidden":
            # Get the index of the last valid hidden state for each sequence
            last_valid_indices = time_mask.sum(dim=1).long() - 1
            pooled = grud_output[torch.arange(grud_output.shape[0]), last_valid_indices]
        elif self.pool == "max":
            pooled = masked_max_pooling(grud_output, time_mask)
        elif self.pool == "mean":
            pooled = masked_mean_pooling(grud_output, time_mask)
        else:
            raise NotImplementedError(f"Pooling function {self.pool} not supported.")

        output = self.output_layer(pooled.to(self.device))
        return output
