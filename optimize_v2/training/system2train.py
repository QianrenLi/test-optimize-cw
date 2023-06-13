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
    def __init__(
        self,
        file_levels: list,
        cw_levels: list,
        memory_size,
        graph: Graph,
        batch_size= 16,
        gamma=0.9,
    ) -> None:
        self.graph = graph
        self.log_path = "./training/logs/log-loss-%s.txt" % time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()
        )
        self.f = open(self.log_path, "a+")
        self.training_counter = 0
        self.is_sorted = False
        self.max_state_num = 3
        self.max_action_num = 5
        self.agent_states_num = 2   # 2 states for each stream

        action_space = []
        action_space.append(file_levels)
        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:        # cw action for each stream
                        action_space.append(cw_levels)

        remain_len = self.max_action_num - len(action_space)
        [action_space.append(cw_levels) for i in range(remain_len)]
        super().__init__(self.max_state_num * self.agent_states_num, action_space, memory_size, batch_size, gamma)

    def update_graph(self, graph):
        self.graph = graph

    def get_state(self):
        state = []
        skipped_state_counter = 0
        active_action = []
        file_action_flag = -1 # 0: not file action, 1: file action   
        # hard embedding
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if (
                        self.graph.info_graph[device_name][link_name][stream_name]["active"]
                        == True
                    ):
                        if "file" not in stream["file_name"]:
                            state.append(stream["rtt"])
                            # state.append(10)
                            state.append(
                                self.graph.info_graph[device_name][link_name][
                                    stream_name
                                ]["target_rtt"]
                            )
                            active_action.append(1)
                            is_cw_action = 1
                        else:
                            file_action_flag = 1
                    else:
                        if "file" not in stream["file_name"]:
                            if not self.is_sorted:
                                state.append(0)
                                state.append(0)
                                active_action.append(-1)                                    # -1 denote action not activated

        active_action.insert(0,file_action_flag)                                            # insert file action space in the first position
        remain_state_len = self.max_state_num * self.agent_states_num - len(state)
        remain_action_len = self.max_action_num - len(active_action)
        [state.append(0) for i in range(remain_state_len)]
        [active_action.append(-1) for i in range(remain_action_len)]

        self.active_action = active_action
        print(active_action)
        return np.array(state)

    def action_to_control(self, state):
        action, action_idx = self.get_action(state)
        # unzip action and action index to a vector
        action = action[0]
        action_idx = action_idx[0]
        control = {}
        idx = 1
        # idx+=1
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    port, tos = stream_name.split("@")
                    
                    if "file" not in stream["file_name"]:
                        if action_idx[idx] != -1:
                            control.update({device_name +"@"+ tos: action[idx]})
                            idx += 1
                        elif not self.is_sorted:
                            idx += 1
                        
        # print(self.active_action)
        control.update({"fraction": action[0]}) if self.active_action[0] != -1 else None
        return control, action_idx

    def get_cost(self, fraction):
        cost = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    target_rtt = self.graph.info_graph[device_name][link_name][
                        stream_name
                    ]["target_rtt"]
                    if (
                        self.graph.info_graph[device_name][link_name][stream_name]["active"]
                        == True
                    ):
                        if target_rtt != 0 and stream["rtt"] is not None:
                            # cost += stream["rtt"] * 1000 > target_rtt
                             cost += abs(stream["rtt"] * 1000 - target_rtt)
        # cost -= fraction
        print("cost",cost)
        return cost


    def training_network(self):
        loss = super().training_network()
        # print("loss",loss)
        self.f.write("{:.6f}\n".format(loss))
        self.training_counter += 1
        return loss


if __name__ == "__main__":
    import test_case as tc
    graph = tc.cw_test_case(50)
    # graph.info_graph["phone"]["wlan_phone_"]["6201@96"]["active"] = False
    graph.show()
    wlanController = wlanDQNController([0.1, 0.2, 0.3], [1, 15], 50, graph)
    # print(wlanController.get_state())
    state = wlanController.get_state()
    action, _ = wlanController.action_to_control(wlanController.get_state())
    print(action)
    fraction = action["fraction"]
    cost = wlanController.get_cost(fraction)
    state_ = wlanController.get_state()
    print(cost)
    wlanController.store_transition(state, _, cost, state_)
    # print(wlanController.active_action)
    # print(wlanController.action_space)
    # print(wlanController.training_network())