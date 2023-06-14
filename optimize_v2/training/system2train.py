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
    '''
    Default the action structure is assumed to be: [[fraction],[cw], [aifs]], where one stream has fraction and the other has. 
    '''
    def __init__(
        self,
        file_levels: list,
        cw_levels: list,
        aifs_levels: list,
        memory_size,
        graph: Graph,
        batch_size=16,
        gamma=0.9,
    ) -> None:
        self.graph = graph
        self.log_path = "./training/logs/log-loss-%s.txt" % time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()
        )
        self.f = open(self.log_path, "a+")
        self.training_counter = 0
        self.is_sorted = False

        self.max_agent_num = 3  # max num of streams
        self.max_action_num = 5  # max num of controllable streams

        self.agent_action_num = 2  # each rtt agent action nums
        self.agent_states_num = 2 + self.agent_action_num  # 4 states for each stream

        self.is_memory_save = True

        action_space = []
        action_space.append(file_levels)
        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:
                        action_space.append(cw_levels)  # cw action for each stream
                        action_space.append(aifs_levels)  # AIFS action for each stream

        remain_len = self.max_action_num * self.agent_action_num + 1  - len(action_space)
        [action_space.append(cw_levels) for i in range(remain_len)]
        super().__init__(
            self.max_state_num * self.agent_states_num,
            action_space,
            memory_size,
            batch_size,
            gamma,
        )

    def update_graph(self, graph):
        self.graph = graph

    def get_state(self):
        state = []
        skipped_state_counter = 0
        active_action = []
        file_action_flag = -1  # -1: not file action, 1: file action
        file_action_value = 0
        # hard embedding
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if (
                        self.graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        ## system state each IC: rtt, target_rtt, cw
                        if "file" not in stream["file_name"]:
                            state.append(stream["rtt"])
                            for _state_name in ["target_rtt", "cw", "aifs"]:
                                state.append(
                                    self.graph.info_graph[device_name][link_name][
                                        stream_name
                                    ][_state_name]
                                )
                            [
                                active_action.append(1)
                                for _ in range(self.agent_action_num)
                            ]
                        else:
                            file_action_flag = 1
                            file_action_value = self.graph.info_graph[device_name][link_name][
                                stream_name
                            ]["throttle"]
                    else:
                        if "file" not in stream["file_name"]:
                            if not self.is_sorted:
                                [state.append(0) for _ in range(self.agent_states_num)]
                                [
                                    active_action.append(-1)
                                    for _ in range(self.agent_action_num)
                                ]  # -1 denote action not activated

        active_action.insert(0, file_action_flag)  # insert file action space in the first position
        state.insert(0, file_action_value)  # insert fraction in the first position
        remain_state_len = self.max_state_num * self.agent_states_num - len(state)
        remain_action_len = self.max_action_num - len(active_action)
        [state.append(0) for i in range(remain_state_len)]
        [active_action.append(-1) for i in range(remain_action_len)]

        self.active_action = active_action
        print(active_action)
        return np.array(state)

    def init_action_guess(self):                            # SOlution for initial action required to training
        action = []
        default_action = {"cw": 3, "aifs": 3, "throttle": 0.6}
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:
                        for _idx, action_name in enumerate(["cw", "aifs"]):
                            self.graph.info_graph[device_name][link_name][
                                stream_name
                            ].update({action_name: default_action[action_name]}) 
                    else:
                        self.graph.info_graph[device_name][link_name][
                            stream_name
                        ].update({"throttle": default_action["throttle"]})                         
    

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
                prot, sender, receiver = link_name.split("_")
                for stream_name, stream in streams.items():
                    port, tos = stream_name.split("@")
                    if "file" not in stream["file_name"]:
                        ## Start Action Load Section
                        _control = {}
                        if action_idx[idx] != -1:
                            for _idx, action_name in enumerate(["cw", "aifs"]):
                                self.graph.info_graph[device_name][link_name][
                                    stream_name
                                ].update({action_name: action[idx + _idx]})
                                _control.update({action_name: action[idx + _idx]})
                            control.update(
                                {prot + "_" + tos + sender + "_" + receiver: _control}
                            )
                            idx += self.agent_action_num
                        elif not self.is_sorted:
                            idx += self.agent_action_num
                        ## End Action Load Section
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
                        self.graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        if target_rtt != 0 and stream["rtt"] is not None:
                            # cost += stream["rtt"] * 1000 > target_rtt
                            cost += abs(stream["rtt"] * 1000 - target_rtt)
        # cost -= fraction
        print("cost", cost)
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

