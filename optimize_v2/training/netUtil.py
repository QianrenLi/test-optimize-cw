# define different network
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class resBlock(nn.Module):
    def __init__(self, kernel_size, channels) -> None:
        # an odd kernel size is recommended
        super().__init__()
        self.conv = self._make_layer(kernel_size, 1, channels, 3)

    def _make_layer(self, kernel_size, in_channels, out_channels , num_layers):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))     
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=False))          

        for i in range(num_layers - 2):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=False))

        layers.append(nn.Conv1d(out_channels, 1, kernel_size, padding=kernel_size // 2))
        layers.append(nn.BatchNorm1d(1))
        layers.append(nn.ReLU(inplace=False))
                  
        return nn.Sequential(*layers)

    def forward(self, x):
        # for i in range(2):
        return self.conv(x) + x


class transBlock(nn.Module):
    def __init__(self, states) -> None:
        # an odd kernel size is recommended
        super().__init__()
        self.query = nn.Linear(states, states)
        self.key = nn.Linear(states, states)
        self.value = nn.Linear(states, states)
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
        return temp_hidden


class ResNet(nn.Module):
    def __init__(self, kernel_size, states, hidden_states, actions):
        super(ResNet, self).__init__()
        self.query = nn.Linear(states, hidden_states)
        self.key = nn.Linear(states, hidden_states)
        self.value = nn.Linear(states, hidden_states)
        self.normal_factor = 1 / np.sqrt(hidden_states)
        self.layer1 = self._make_layer(kernel_size, hidden_states, 2)
        self.fc = nn.Linear(hidden_states, actions)

    def _make_layer(self, kernel_size, hidden_states, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(transBlock(hidden_states))
            layers.append(resBlock(kernel_size, 3))
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

        return self.fc(temp_hidden)
