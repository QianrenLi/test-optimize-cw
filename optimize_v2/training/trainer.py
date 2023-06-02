import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# define network


class Net(nn.Module):
    def __init__(self, states, hidden_states, actions):
        super(Net, self).__init__()
        self.query = nn.Linear(states, hidden_states)
        self.key = nn.Linear(states, hidden_states)
        self.value = nn.Linear(states, hidden_states)
        self.fc = nn.Linear(hidden_states, actions)
        self.normal_factor = 1/np.sqrt(hidden_states)

    def forward(self, x):
        temp_q = self.query(x)
        temp_k = self.key(x)
        temp_v = self.value(x)

        self.attention_weights = nn.Softmax(dim=-1)(torch.bmm(
            temp_q, temp_k.permute(0, 2, 1)))*self.normal_factor

        temp_hidden = torch.bmm(self.attention_weights, temp_v)

        return F.relu(self.fc(temp_hidden))


# define the controller for training
class DQNController():
    def __init__(self, state_size, action_space: list, memory_size, batch_size=4, gamma=0.9) -> None:
        self.state_size = state_size
        self.action_space = action_space
        self.action_num = len(action_space)
        self.action_size = sum([len(i) for i in action_space])
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = np.zeros(
            (memory_size, state_size * 2 + self.action_num + 1))
        self.eval_net = Net(state_size, state_size * 2, self.action_size)
        self.action_net = Net(state_size, state_size * 2, self.action_size)
        self.eval_opt = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def set_network(self, eval_net: nn.Module, action_net: nn.Module):
        self.eval_net = eval_net
        self.action_net = action_net

    def set_opt(self, eval_opt):
        self.eval_opt = eval_opt

    def set_criterion(self, criterion):
        self.criterion = criterion

    def _extract_action(self, batch_size ,tensor_action):
        np_extracted_action = np.zeros(
            (batch_size, self.action_num), dtype=np.int32)
        np_action_index = np.zeros(
            (batch_size, self.action_num), dtype=np.int32)

        for batch_id in range(batch_size):
            pointer = 0
            for i in range(self.action_num):
                np_extracted_action[batch_id, i] = tensor_action[batch_id,
                                                                 0, pointer:pointer+len(self.action_space[i])].argmin()
                np_action_index[batch_id,
                                i] = np_extracted_action[batch_id, i] + pointer
                np_extracted_action[batch_id, i] = self.action_space[i][int(
                    np_extracted_action[batch_id, i])]
                pointer += len(self.action_space[i])
        return np_extracted_action, np_action_index

    def get_action(self, state):
        print('state',state)
        state = torch.tensor(state.reshape((1,1,self.state_size)), dtype= torch.float)
        actions = self.action_net(state).detach().numpy()
        if np.random.rand() <= 0.9:
            self.action, action_idx = self._extract_action(1,actions)
        else:
            index = list(np.random.randint([len(i) for i in self.action_space]))
            action_idx = np.array([index])
            self.action = np.array([[self.action_space[i][index[i]] for i in range(len(index))]])
        return self.action, action_idx

    def store_transition(self, state, action, cost, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, action, cost, state_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def extract_memory(self, batch_memory):
        state = batch_memory[:, :self.state_size]
        action = batch_memory[:,
                              self.state_size: self.state_size + self.action_num]
        cost = batch_memory[:, self.state_size +
                            self.action_num:self.state_size + self.action_num + 1]
        state_ = batch_memory[:, self.state_size + self.action_num + 1:]
        return state, action, cost, state_

    @staticmethod
    def tensor_formatting(np_array, ts_shape, dtype):
        return torch.tensor(np_array.reshape(ts_shape), dtype=dtype)

    def _action_tensor_formatting(self, cost, action_tensor, action_idx: np.ndarray):
        action_idx = torch.tensor(action_idx.reshape(
            self.batch_size, 1, self.action_num), dtype=torch.int64)
        return torch.tensor(np.repeat(cost, self.action_num, axis=1).reshape(self.batch_size, 1, self.action_num), dtype=torch.float) \
            + self.gamma * action_tensor.gather(2, action_idx).clone().detach()

    def training_network(self):
        if None in [self.eval_opt, self.criterion, self.eval_net]:
            print("Check optimizer, criterion and network setup")
            return
        # extract memory
        index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[index, :]
        state, action, cost, state_ = self.extract_memory(batch_memory)
        state = self.tensor_formatting(
            state, (self.batch_size, 1, self.state_size), torch.float)
        action = self.tensor_formatting(
            action, (self.batch_size, 1, self.action_num), torch.int64)

        state_ = self.tensor_formatting(
            state_, (self.batch_size, 1, self.state_size), torch.float)
        ## training in batch
        q_eval = self.eval_net(state).gather(2, action)
        q_action_next = self.action_net(state).detach()

        _, action_idx = self._extract_action(self.batch_size,q_action_next)
        q_target = self._action_tensor_formatting(
            cost, q_action_next, action_idx)
        # zero gradient
        self.eval_opt.zero_grad()

        # back propagate
        loss = self.criterion(q_eval, q_target)
        loss.backward()

        # update network
        self.eval_opt.step()
        return loss.item()

    def parameter_replace(self):
        for action_param, param in zip(self.action_net.parameters(), self.eval_net.parameters()):
            action_param.data.copy_(param.data)


if __name__ == '__main__':
    controller = DQNController(10, [[1, 2, 3], [1, 2]], 50)
    # eval_net = Net()

    print("{:.8f}".format(controller.training_network()))
