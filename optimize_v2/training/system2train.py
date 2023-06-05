import sys
import os


import numpy as np
import torch
import time

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from transmission_graph import Graph
from training.trainer import DQNController


class wlanDQNController(DQNController):
    def __init__(self, file_levels:list, cw_levels:list, memory_size, graph:Graph, batch_size=4, gamma=0.9) -> None:
        self.graph = graph
        self.log_path = f"./training/logs/log-loss{time.ctime()}.txt"
        self.f = open(self.log_path, 'a+')
        state_size = 0
        self.training_counter = 0
        action_space = []
        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    state_size += 1
                    if "file" in stream["file_name"]:
                        action_space.append(file_levels)
                    else:
                        action_space.append(cw_levels)
        super().__init__(state_size, action_space, memory_size, batch_size, gamma)

    def update_graph(self,graph):
        self.graph = graph


    def get_state(self):
        state = []
        throughput = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:
                        state.append(stream["rtt"] * 1000)
                    throughput += stream["rtt"] * 1000
        state.append(throughput) #TODO: let the throttle to be the system state
        return np.array(state)


    def action_to_control(self,state):
        action,action_idx = self.get_action(state)
        ## unzip action and action index to a vector
        action = action[0]
        action_idx = action_idx[0]
        # print("action",action)
        control = {}
        idx = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" in stream["file_name"]:
                        control.update({"fraction":action[idx]})
                        # idx += 1
                    else:
                        control.update({device_name:action[idx]})       
                    idx += 1
        return control,action_idx


    def get_cost(self,fraction):
        cost = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    target_rtt = self.graph.info_graph[device_name][link_name][stream_name]["target_rtt"]
                    if target_rtt != 0:
                        cost += abs(stream["rtt"] * 1000 - target_rtt)
                    # cost += stream["rtt"] * 1000
        # cost -= fraction * 60
        return cost

    def store_params(self,path):
        torch.save(self.eval_net.state_dict(), path)

    def training_network(self):
        loss = super().training_network()
        self.f.write("{:.6f}\n".format(loss))
        self.training_counter += 1
        return loss

if __name__ == '__init__':
    wlanController = wlanDQNController([1,2,3],[1,2],50,Graph())

