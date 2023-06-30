from trainer import DQNController
import numpy as np
from torchsummary import summary
import torch
from torch import nn
import os
abs_path = os.path.dirname(os.path.realpath(__file__))


gamma = 0.9
agent_num = 4
controller = DQNController(agent_num,[[-2,-1,0,1,2] for _ in range(agent_num)], 10000, 64, gamma)
# controller.active_action = [1,1,-1]
controller.train_counter_max = 200

def state_to_cost(state):
    a = sum(abs(state))
    return a

def action_to_state(state, action):
    # print(action)
    # state_ = state + action + 0.5 * np.random.rand(len(state))
    state_ = np.zeros(agent_num)
    for i in range(len(state)):
        state_[i] = max(min(state[i] + action[i] + 0.5 * np.random.randint(2), 50), -50) if controller.active_action[i] != -1 else 0
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


def evaluation(controller, state_num):
    episode = 200
    state = np.zeros(state_num)
    value = 0
    for i in range(episode):
        # state = np.array([50])
        action, action_idx = controller.get_action(state)
        action = action[0]
        action_idx = action_idx[0]

        state_ = action_to_state(state, action)
        cost = state_to_cost(state_)
        controller.store_transition(state , action_idx, cost, state_)
        state = state_

        value = cost + value * gamma

    print("Evaluation value",value)

state_ = None
value = 0
value_list = []
action_list = []
state_list = []
counter_max = 100
counter = 0
loss_saved = 0
episode = 1000
num_episode = 50
# controller.load_params("%s/model/param.pkl" % abs_path)

for i_episode in range(num_episode):
    # controller.action_counter = 0
    state = np.zeros(agent_num)
    state_ = None    
    value = 0
    loss = 0
    for i in range(episode):
        action, action_idx = controller.get_action(state)
        action = action[0]
        action_idx = action_idx[0]

        state_ = action_to_state(state, action)
        cost = state_to_cost(state_)
        controller.store_transition(state , action_idx, cost, state_)
        state = state_
        if controller.memory_counter > 100:
            loss = controller.training_network()

        value = cost + value * gamma
        action_list.append(action[0])
        state_list.append(state[0])

        if counter < counter_max:
            loss_saved += loss 
        else:
            print("loss\t", loss_saved/counter_max)   
            counter = 0
            loss_saved = 0
        counter += 1 

    evaluation(controller, agent_num)
    controller.store_params("%s/model/param.pkl" % abs_path)
    print("value", value)
    value_list.append(value)
    

import matplotlib.pyplot as plt

plt.plot(value_list)
plt.show()

plt.plot(action_list)
plt.show()

plt.plot(state_list)
plt.show()