import torch.nn as nn
import torch


class Network(nn.Module):
    """Standard MLP Q-network."""

    def __init__(self, input_size, action_space, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.relu = nn.ReLU()
        layer_sizes = [input_size] + hidden_sizes

        self.input_layer = nn.Linear(layer_sizes[0], layer_sizes[1])

        # A nice way to create a custom number of layers is to use ModuleList
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(1, len(layer_sizes) - 1)]
        )

        # define our output layer
        self.outl = nn.Linear(layer_sizes[-1], action_space)

    def forward(self, x):
        # first output is from our input layer (feature extraction)
        out = self.relu(self.input_layer(x))

        # go through our hidden layers
        for layer in self.hidden_layers:
            out = self.relu(layer(out))

        # return result after passing through our output layer
        return self.outl(out)


class DuelingNet(nn.Module):
    """Dueling Net"""

    def __init__(self, input_size, action_space,
                 trunk_sizes=None,
                 value_sizes=None,
                 advantage_sizes=None):
        super().__init__()
        if advantage_sizes is None:
            advantage_sizes = [128]
        if value_sizes is None:
            value_sizes = [128]
        if trunk_sizes is None:
            trunk_sizes = [256, 256]
        self.relu = nn.ReLU()

        # --- Trunk ---
        trunk_dims = [input_size] + trunk_sizes
        self.trunk = nn.ModuleList(
            [nn.Linear(trunk_dims[i], trunk_dims[i + 1]) for i in range(len(trunk_dims) - 1)]
        )

        trunk_out = trunk_sizes[-1]

        # --- Value head --- outputs V(s): scalar
        self.value_head = self._build_head(trunk_out, value_sizes, out_size=1)

        # --- Advantage head --- outputs A(s,a): vector of length action_space
        self.advantage_head = self._build_head(trunk_out, advantage_sizes, out_size=action_space)

    def _build_head(self, in_size, hidden_sizes, out_size):
        dims = [in_size] + hidden_sizes + [out_size]
        return nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def _forward_layers(self, x, layers, apply_relu_to_last=False):
        for i, layer in enumerate(layers):
            is_last = (i == len(layers) - 1)
            x = layer(x) if (is_last and not apply_relu_to_last) else self.relu(layer(x))
        return x

    def forward(self, x):
        trunk = self._forward_layers(x, self.trunk, apply_relu_to_last=True)
        V = self._forward_layers(trunk, self.value_head)
        A = self._forward_layers(trunk, self.advantage_head)
        return V + A - A.mean(dim=1, keepdim=True)


class REINFORCE_Network(nn.Module):
    def __init__(self, input_size, action_space, hidden_sizes=None, discrete: bool = False):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        self.discrete = discrete
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        layer_sizes = [input_size] + hidden_sizes
        self.input_layer = nn.Linear(layer_sizes[0], layer_sizes[1])

        self.hidden_layers = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(1, len(layer_sizes) - 1)
        )

        self.out_layer_discrete = nn.Linear(layer_sizes[-1], action_space)
        # each action has its own mean and std to sample from
        self.out_layer_continuous = nn.Linear(layer_sizes[-1], action_space * 2)

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.relu(layer(out))

        if self.discrete:
            out = self.softmax(self.out_layer_discrete(out))
            return out

        else:
            mean, log_std = out.chunk(2, dim=-1)
            mean = torch.tanh(mean)  # squish to (-1, 1)
            std = log_std.exp().clamp(1e-3, 1.0)  # must be positive
            return mean, std

