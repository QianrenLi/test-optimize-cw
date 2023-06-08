from trainer import DQNController
import numpy as np
from torchsummary import summary
import torch
from torch import nn
from netUtil import ResNet
gamma = 0.9
controller = DQNController(2,[[-5,-1,0,1,5],[-5,-1,0,1,5]], 10000, 16, gamma)
# action_net = ResNet(
#     3, controller.state_size, controller.state_size * 50, controller.action_size
# )
# eval_net = ResNet(
#     3, controller.state_size, controller.state_size * 50, controller.action_size
# )
# action_opt = torch.optim.Adam(action_net.parameters(), lr=0.0001)
# controller.set_network(action_net, action_net, action_opt)
controller.train_counter_max = 100
# summary(controller.eval_net, (1,2), -1 , device="cpu")

def state_to_cost(state):
    a = sum(abs(state - np.array([0,0])))
    return a

def action_to_state(state, action):
    state_ = state + action - 2
    for i in range(len(state_)):
        state_[i] = max(min(state_[i], 50), -50)
    return state_


def see_param(model:nn.Module):
    saved_dict = {}
    for name,param in model.named_parameters():
        saved_dict.update({name: param.clone().detach().numpy()})
        # print(saved_dict[name])
    # print(saved_dict)
    return saved_dict

def compute_weight_diff(params, params_):
    weight_diff = 0
    for name in params:
        weight_diff += np.sum(np.abs(params[name] - params_[name]))
    return weight_diff


state_ = None
value = 0
value_list = []
action_list = []
state_list = []
counter_max = 100
counter = 0
loss_saved = 0
episode = 500
num_episode = 10

for i_episode in range(num_episode):
    # controller.action_counter = 0
    state = np.array([0,0])
    state_ = None    
    value = 0
    for i in range(episode):
        if state_ is not None:
            state = state_
        action, action_idx = controller.get_action(state)
        action = action[0]
        action_idx = action_idx[0]
        cost = state_to_cost(state)
        state_ = action_to_state(state, action)

        # print(state, state_,cost, action,action_idx)
        controller.store_transition(state , action_idx, cost, state_)
        # print(controller.memory)
        # params = see_param(controller.action_net)
        loss = controller.training_network()
        # params_ = see_param(controller.action_net)
        # weight_diff = compute_weight_diff(params,params_)
        value = cost + value * gamma
        action_list.append(action[0])
        state_list.append(state[0])
        # print(action)
        # print(weight_diff) if weight_diff != 0 else None
        if counter < counter_max:
            loss_saved += loss 
        else:
            print("loss\t", loss_saved)   
            counter = 0
            loss_saved = 0
        counter += 1 
    print("value", value)
    value_list.append(value)
    

import matplotlib.pyplot as plt

plt.plot(value_list)

plt.show()

plt.plot(action_list)
plt.show()

plt.plot(state_list)
plt.show()