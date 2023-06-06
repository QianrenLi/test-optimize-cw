# define different network
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class resBlock(nn.Module):
    def __init__(self, kernel_size) -> None:
        # an odd kernel size is recommended
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        for i in range(2):
            x = self.conv1(x)
        return F.relu(self.conv1(x) + x)


class transBlock(nn.Module):
    def __init__(self, states, kernel_size) -> None:
        # an odd kernel size is recommended
        super().__init__()
        self.query = nn.Linear(states, states)
        self.key = nn.Linear(states, states)
        self.value = nn.Linear(states, states)
        self.conv1 = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)
        self.normal_factor = 1 / np.sqrt(states)

    def forward(self, x):
        temp_q = self.query(x)
        temp_k = self.key(x)
        temp_v = self.value(x)

        self.attention_weights = (
            nn.Softmax(dim=-1)(torch.bmm(temp_q, temp_k.permute(0, 2, 1)))
            * self.normal_factor
        )

        temp_hidden = torch.bmm(self.attention_weights, temp_v)
        return F.relu(self.conv1(temp_hidden) + x)


class ResNet(nn.Module):
    def __init__(self, kernel_size, states, hidden_states, actions):
        super(ResNet, self).__init__()
        self.query = nn.Linear(states, hidden_states)
        self.key = nn.Linear(states, hidden_states)
        self.value = nn.Linear(states, hidden_states)
        self.normal_factor = 1 / np.sqrt(hidden_states)
        self.layer1 = self._make_layer(kernel_size, hidden_states, 5)
        self.fc = nn.Linear(hidden_states, actions)

    def _make_layer(self, kernel_size, hidden_states, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(transBlock(hidden_states, kernel_size))
            layers.append(resBlock(kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        temp_q = self.query(x)
        temp_k = self.key(x)
        temp_v = self.value(x)

        self.attention_weights = (
            nn.Softmax(dim=-1)(torch.bmm(temp_q, temp_k.permute(0, 2, 1)))
            * self.normal_factor
        )

        temp_hidden = torch.bmm(self.attention_weights, temp_v)

        temp_hidden = self.layer1(temp_hidden)

        return F.relu(self.fc(temp_hidden))
