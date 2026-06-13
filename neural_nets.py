import torch.nn as nn
import torch
import numpy as np

class Network(nn.Module):
    """Standard MLP Q-network."""

    def __init__(self, input_size, action_space, hidden_sizes=None, output_activation: bool = False):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.output_activation = output_activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
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
        if not self.output_activation:
            return self.outl(out)
        else:
            return self.tanh(self.outl(out))


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


class ContinuousNetwork(nn.Module):
    def __init__(self, input_size, action_space, hidden_sizes=None, log_init: float = -1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.activation = nn.Tanh()

        # Build the network trunk
        layer_sizes = [input_size] + hidden_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(self.activation)
        self.trunk = nn.Sequential(*layers)

        self.action_mean = nn.Linear(hidden_sizes[-1], action_space)

        # nn.Parameter is literally a stand-alone set of tensors that can tweakable
        #self.action_logstd = nn.Parameter(torch.zeros(action_space))
        self.action_logstd = nn.Parameter(torch.full((action_space,), -1.0))

        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization makes a massive difference in early training stability
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Scale down the output layer weight so initial actions are near 0
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, 0.0)

    def forward(self, x):
        hidden_features = self.trunk(x)
        mean = self.action_mean(hidden_features)

        # expand log_std to match the batch dimensions of the mean
        # this means from (,6) to (batch_size, 6)
        std = torch.exp(self.action_logstd).expand_as(mean)

        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.relu = nn.ReLU()

        input_size = state_dim + action_dim
        layer_sizes = [input_size] + hidden_sizes

        self.input_layer = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(1, len(layer_sizes) - 1)]
        )
        # Outputs a single Q-value evaluation
        self.outl = nn.Linear(layer_sizes[-1], 1)

    def forward(self, state, action):
        # Concatenate the state and action tensors side-by-side
        # If state is [batch_size, 27] and action is [batch_size, 8]
        # x will become [batch_size, 35]
        x = torch.cat([state, action], dim=1)

        # first output is from our input layer (feature extraction)
        out = self.relu(self.input_layer(x))

        # go through our hidden layers
        for layer in self.hidden_layers:
            out = self.relu(layer(out))

        # return result after passing through our output layer
        return self.outl(out)


