import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# define network


class Net(nn.Module):
    def __init__(self, states, hidden_states, actions):
        super(Net, self).__init__()
        self.query = nn.Linear(states, hidden_states)
        # self.key = nn.Linear(states, hidden_states)
        # self.value = nn.Linear(states, hidden_states)

        # self.normal_factor = 1 / np.sqrt(hidden_states)

        self.fc = self._make_layer(hidden_states, 2)
        self.fc1 = nn.Linear(hidden_states, actions)

    def _make_layer(self, hidden_states, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_states, hidden_states))
            layers.append(nn.BatchNorm1d(1))
            layers.append(nn.ReLU(inplace=False))
            # layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    def forward(self, x):
        temp_q = self.query(x)
        # temp_k = self.key(x)
        # temp_v = self.value(x)

        # self.attention_weights = (
        #     nn.Softmax(dim=-1)(torch.bmm(temp_q, temp_k.permute(0, 2, 1)))
        #     * self.normal_factor
        # )

        # temp_hidden = torch.bmm(self.attention_weights, temp_v)

        return self.fc1(self.fc(temp_q))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# define the controller for training
class DQNController:
    def __init__(
        self,
        state_size,
        action_space: list,
        memory_size,
        batch_size=4,
        gamma=0.9,
        hidden_state=1024,
    ) -> None:
        self.state_size = state_size
        self.action_space = action_space
        self.action_num = len(action_space)
        self.active_action = [1 for i in range(self.action_num)]
        self.action_size = sum([len(i) for i in action_space])
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.action_counter = 0
        self.train_counter = 0
        self.memory_counter = 0
        self.train_counter_max = 1000

        self.is_memory_save = False

        self.memory = np.zeros((memory_size, state_size * 2 + self.action_num + 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = Net(state_size, hidden_state, self.action_size).to(device)
        self.action_net = Net(state_size, hidden_state, self.action_size).to(device)
        self.action_opt = torch.optim.Adam(self.action_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss().to(device)

        self.parameter_replace()

    def load_params(self, path):
        self.action_net.load_state_dict(torch.load(path))
        self.parameter_replace()

    def store_params(self, path):
        torch.save(self.action_net.state_dict(), path)

    def load_memory(self, path:str):
        _memory = np.load(path)
        self.memory_counter = len(_memory)
        self.memory_size = self.memory_counter
        self.memory = _memory

    def store_memory(self,  path:str):
        memory_limit = min(self.memory_size, self.memory_counter)
        _memory = self.memory[: memory_limit]
        np.save(path, _memory)

    def set_network(
        self,
        eval_net: nn.Module,
        action_net: nn.Module,
        action_opt: torch.optim.Optimizer,
    ):
        self.eval_net = eval_net
        self.action_net = action_net
        self.action_opt = action_opt

    def set_opt(self, action_opt):
        self.action_opt = action_opt

    def set_criterion(self, criterion):
        self.criterion = criterion

    def extract_memory(self, batch_memory: np.array):
        state = batch_memory[:, : self.state_size]
        action = batch_memory[:, self.state_size : self.state_size + self.action_num]
        cost = batch_memory[
            :, self.state_size + self.action_num : self.state_size + self.action_num + 1
        ]
        state_ = batch_memory[:, self.state_size + self.action_num + 1 :]
        return state, action, cost, state_

    def _extract_action(self, tensor_action, action_idxs):
        """
        function used to extract action and action indicator from action tensor
        tensor_action (batch_size, 1, action_size)
        action_idxs (batch_size, active_action_num)
        """
        batch_size = len(tensor_action)
        active_action_num = len(action_idxs[0])
        np_extracted_action = np.zeros((batch_size, active_action_num))
        np_action_index = np.zeros((batch_size, active_action_num), dtype=np.int32)

        for batch_id, action_idx in enumerate(action_idxs):
            pointer = 0
            for i in range(len(action_idx)):
                if action_idx[i] != -1:  # skip action if action is not active
                    np_extracted_action[batch_id, i] = tensor_action[
                        batch_id, 0, pointer : pointer + len(self.action_space[i])
                    ].argmin()
                    np_action_index[batch_id, i] = (
                        np_extracted_action[batch_id, i] + pointer
                    )
                    np_extracted_action[batch_id, i] = self.action_space[i][
                        int(np_extracted_action[batch_id, i])
                    ]
                else:
                    np_extracted_action[
                        batch_id, i
                    ] = -1  # set action to -1 if action is not active
                    np_action_index[batch_id, i] = -1
                pointer += len(self.action_space[i])
        return np_extracted_action, np_action_index

    def store_transition(self, state, action_idx, cost, state_):
        transition = np.hstack((state, action_idx, cost, state_))
        if None in transition:
            print(transition)
            return
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        if index == 0 and self.is_memory_save:
            self.store_memory()
        self.memory_counter += 1
        

    def get_action(self, state):
        """
        function used for output action to environment when given a state
        """
        epsilon = min(self.action_counter / 100, 0.9)
        if np.random.rand() <= epsilon:
            state = torch.tensor(
                state.reshape((1, 1, self.state_size)), dtype=torch.float
            ).to(device)
            actions = self.action_net(state).cpu().clone().detach().numpy()
            # print(actions)
            self.action, action_idx = self._extract_action(
                actions, np.array([self.active_action])
            )
        else:
            index = list(
                [
                    np.random.randint(len(self.action_space[i]))
                    if self.active_action[i] != -1
                    else -1
                    for i in range(len(self.action_space))
                ]
            )
            action_idx = np.array([index])
            # print(action_idx)
            self.action = np.array(
                [[self.action_space[i][index[i]] for i in range(len(index))]]
            )
        self.action_counter += 1
        return self.action, action_idx

    def _depart_batch(self, action_idxs: np.array):
        """
        depart the batch based on active actions
        """
        # action_idx = []
        active_action_nums = []
        active_action_batches = []
        for batch_idx, action_idx in enumerate(action_idxs):
            active_action_num = sum(action_idx > -1)
            if active_action_num in active_action_nums:
                _idx = active_action_nums.index(active_action_num)
                active_action_batches[_idx].append(batch_idx)
            else:
                active_action_nums.append(active_action_num)
                active_action_batches.append([batch_idx])
        return active_action_batches

    def _remove_inactive_action(self, action_idxs):
        """
        remove inactive action from action index from same batch length
        """
        active_action_idxs = []
        for action_idx in action_idxs:
            active_action_idxs.append(action_idx[action_idx > -1])
        return np.array(active_action_idxs)

    @staticmethod
    def tensor_formatting(np_array, ts_shape, dtype):
        return torch.tensor(np_array.reshape(ts_shape), dtype=dtype).to(device)

    def _action_tensor_formatting(self, cost, action_tensor, action_idx: np.ndarray):
        batch_size = len(action_tensor)
        action_num = len(action_idx[0])
        # print(action_idx)
        action_idx = torch.tensor(
            action_idx.reshape(batch_size, 1, action_num), dtype=torch.int64
        ).to(device)
        return torch.tensor(
            np.repeat(cost, action_num, axis=1).reshape(batch_size, 1, action_num),
            dtype=torch.float,
        ).to(device) + self.gamma * action_tensor.gather(2, action_idx)

    def parameter_replace(self):
        # print("pre\t",self.action_net.state_dict())
        self.eval_net.load_state_dict(self.action_net.state_dict())
        # print("a\t",self.action_net.state_dict())

    def training_network(self):
        if None in [self.action_opt, self.criterion, self.eval_net]:
            print("Check optimizer, criterion and network setup")
            return
        # extract memory
        index = np.random.choice(
            min(self.memory_size, self.memory_counter), self.batch_size
        )
        batch_memory = self.memory[index, :]
        state, action_idx, cost, state_ = self.extract_memory(batch_memory)
        # select batch, addressing the different active action
        active_action_batches = self._depart_batch(action_idx)
        for batch_idx in active_action_batches:
            _state = self.tensor_formatting(
                np.take(state, batch_idx, axis=0), (-1, 1, self.state_size), torch.float
            )

            _action_idx_np = np.take(action_idx, batch_idx, axis=0)
            _action_idx = self._remove_inactive_action(_action_idx_np)
            _action_idx = self.tensor_formatting(
                _action_idx,
                (-1, 1, _action_idx.shape[1]),
                torch.int64,
            )
            _state_ = self.tensor_formatting(
                np.take(state_, batch_idx, axis=0),
                (-1, 1, self.state_size),
                torch.float,
            )
            _cost = np.take(cost, batch_idx, axis=0)
            q_action = self.action_net(_state)
            q_action = q_action.gather(2, _action_idx)

            with torch.no_grad():
                q_action_next = self.eval_net(_state_).detach()
                _, _action_idx = self._extract_action(q_action_next, _action_idx_np)
                q_target = self._action_tensor_formatting(
                    _cost, q_action_next, self._remove_inactive_action(_action_idx)
                )
            self.train_counter += 1

            self.action_opt.zero_grad()
            # back propagate
            loss = self.criterion(q_action, q_target)

            # zero gradient
            loss.backward()
            # set allowable maximum gradient
            torch.nn.utils.clip_grad_value_(self.action_net.parameters(), 1000)
            # update network
            self.action_opt.step()

        if self.train_counter >= self.train_counter_max:
            self.train_counter = 0
            self.parameter_replace()

        return loss.item()


if __name__ == "__main__":
    controller = DQNController(10, [[1, 2, 3], [1, 2]], 50)

    # from netUtil import ResNet

    # action_net = ResNet(
    #     3, controller.state_size, controller.state_size * 100, controller.action_size
    # )
    # eval_net = ResNet(
    #     3, controller.state_size, controller.state_size * 100, controller.action_size
    # )
    # action_opt = torch.optim.Adam(eval_net.parameters(), lr=0.0001)
    # controller.set_network(action_net, action_net, action_opt)

    print("{:.8f}".format(controller.training_network()))
