import torch.nn as nn

class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2, dropout=0.1):
        super().__init__()
        layers, dims = [], [input_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            layers += [nn.Linear(dims[i], dims[i + 1]),
                       nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x: (B, L, D)
        return self.net(x)                      # (B, L, hid_dim)