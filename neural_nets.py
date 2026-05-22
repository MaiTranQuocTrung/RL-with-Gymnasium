import torch.nn as nn

class Network(nn.Module):
    """Standard MLP Q-network."""

    def __init__(self, input_size, action_space, hidden_size=256, n_layers=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(input_size, hidden_size)

        # A nice way to create a custom number of layers is to use ModuleList
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)] # -1 since remember the first layer is input layer
        )

        # define our output layer
        self.outl = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        # first output is from our input layer (feature extraction)
        out = self.relu(self.input_layer(x))

        # go through our hidden layers
        for layer in self.hidden_layers:
            out = self.relu(layer(out))

        # return result after passing through our output layer
        return self.outl(out)


class DuelingNet(nn.Module):
    """Dueling DQN"""

    def __init__(self, input_size, action_space, hidden_size=256):
        super().__init__()

        # This is the common "trunk" of the network, or feature extraction
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # This head should output a single float32 (maybe switch to 64?) defined as V(s) = value of state independent of actions
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # This head should output a vector of length action_space
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        # feature extrction
        trunk = self.trunk(x)

        # Get V(s)
        V = self.value_head(trunk)

        # Get A(s)
        A = self.advantage_head(trunk)
        return V + A - A.mean(dim=1, keepdim=True)  # mean subtraction for relative contribution of each actions to Q(s,a), dim=1 since its a vector


