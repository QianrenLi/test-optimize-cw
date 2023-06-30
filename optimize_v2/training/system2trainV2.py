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
        max_agent_num = 4,
        max_action_num = 5,
        is_CDQN = False,
        is_k_cost = 0
    ) -> None:
        self.graph = graph
        self.log_path = "./training/logs/log-loss-%s.txt" % time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()
        )
        self.f = open(self.log_path, "a+")
        self.training_counter = 0
        self.is_sorted = False

        self.max_agent_num = max_agent_num         # max num of streams
        self.max_action_num = max_action_num       # max num of controllable streams

        self.agent_action_num = 2           # each rtt agent action nums
        self.agent_states_num = 3           # 3 states for each stream

        self.is_memory_save = True
        self.is_action_threshold = True         # Action threshold for outlier state

        self._rtt_threshold = 0.04                      # In (s), constraint state
        self._cost_threshold = 100               # Clip the cost

        self.cw_levels = cw_levels
        self.aifs_levels = aifs_levels
        self.file_levels = file_levels

        action_space = []
        action_space.append([-2, -1, 0, 1, 2])
        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if "file" not in stream["file_name"]:
                        action_space.append([-2, -1, 0, 1, 2])  # cw difference action for each stream
                        action_space.append([-2, -1, 0, 1, 2])  # AIFS difference action for each stream

        remain_len = ((self.max_action_num * self.agent_action_num + 1  - len(action_space))) // 2
        [[action_space.append([-2, -1, 0, 1, 2]), action_space.append([-2, -1, 0, 1, 2])] for i in range(remain_len)]
        super().__init__(
            self.max_agent_num * self.agent_states_num + 1,
            action_space,
            memory_size,
            batch_size,
            gamma,
            is_CDQN = is_CDQN,
            is_k_cost = is_k_cost,
            is_remove_state_maximum = self._rtt_threshold
        )

    def update_graph(self, graph):
        self.graph = graph

    def get_state(self):
        """
        Get the state of the system;
        From system state conclude corresponding active action -> how many rtt stream, is file stream exist.
        """
        state = []
        skipped_state_counter = 0
        active_action = []
        file_action_flag = -1  # -1: not file action, 1: file action
        his_fraction = 0
        # hard embedding
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                is_rtt_active = False
                for stream_name, stream in streams.items():
                    if (
                        self.graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        ## system state each IC: rtt, target_rtt, cw
                        if "file" not in stream["file_name"]:
                            ##
                            is_rtt_active = True
                            # state.append(10)
                            if stream["rtt"] < self._rtt_threshold:
                                state.append(stream["rtt"] * 1000)
                            else:
                                state.append(self._rtt_threshold * 1000)

                            for _state_name in ["cw", "aifs"]:
                                state.append(
                                    self.graph.info_graph[device_name][link_name][
                                        stream_name
                                    ][_state_name]
                                )
                            # 
                            [
                                active_action.append(1)
                                for _ in range(self.agent_action_num)
                            ]
                        else:
                            file_action_flag = 1                    # -1 denote action not activated
                            his_fraction = self.graph.info_graph[device_name][link_name][
                                        stream_name
                                    ]["fraction"]

                if not is_rtt_active and not self.is_sorted:
                    [state.append(0) for _ in range(self.agent_states_num)]
                    [
                        active_action.append(-1)
                        for _ in range(self.agent_action_num)
                    ] 

        active_action.insert(0, file_action_flag)  # insert file action space in the first position
        # print(active_action)
        remain_state_len = self.max_agent_num * self.agent_states_num - len(state)
        remain_action_len = (self.max_action_num * self.agent_action_num + 1 - len(active_action)) // 2
        [state.append(0) for i in range(remain_state_len)]
        state.append(his_fraction)
        [[active_action.append(-1) for _i in range(self.agent_action_num)] for i in range(remain_action_len)]

        self.active_action = active_action
        return np.array(state)

    def init_action_guess(self):                            # Solution for initial action required to training
        default_action = {"cw": 3, "aifs": 3, "fraction": 0.6}
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
                        ].update({"fraction": default_action["fraction"]})                         
    
    def update_action_from_history(self, delta_action, his_action, keyword = "cw"):
        """
        Update action from history action
        """
        if keyword == "cw":
            action_list = self.cw_levels
        elif keyword == "aifs":
            action_list = self.aifs_levels
        else:
            action_list = self.file_levels
        fir_idx = np.where(np.array(action_list) == his_action)[0][0]
        print("delta_action", delta_action)
        sec_idx = fir_idx + delta_action if fir_idx + delta_action < len(action_list) else len(action_list) - 1
        sec_idx = 0 if sec_idx < 0 else sec_idx

        return action_list[int(sec_idx)]

    def action_to_control(self, state):
        action, action_idx = self.get_action(state)
        # print(action)
        is_state_reach_maximum =  True  if self.is_remove_state_maximum in state else False
        # unzip action and action index to a vector
        action = action[0]
        action_idx = action_idx[0]
        control = {}
        idx = 1
        # idx+=1
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                is_rtt_active = False
                prot, sender, receiver = link_name.split("_")
                for stream_name, stream in streams.items():
                    port, tos = stream_name.split("@")
                    if "file" not in stream["file_name"]:
                        is_rtt_active = True
                        ## Start Action Load Section
                        _control = {}
                        if action_idx[idx] != -1:
                            for _idx, action_name in enumerate(["cw", "aifs"]):
                                his_action = self.graph.info_graph[device_name][link_name][stream_name][action_name]
                                if self.is_action_threshold and is_state_reach_maximum:
                                    next_action = self.update_action_from_history(-1, his_action, action_name)
                                    self.graph.info_graph[device_name][link_name][
                                        stream_name
                                    ].update({action_name: next_action})
                                    _control.update({action_name: next_action})
                                else:
                                    next_action = self.update_action_from_history(action[idx + _idx], his_action, action_name)
                                    self.graph.info_graph[device_name][link_name][
                                        stream_name
                                    ].update({action_name: next_action})
                                    _control.update({action_name: next_action})
                            control.update(
                                {prot + "_" + tos + "_" + sender + "_" + receiver: _control}
                            )
                            idx += self.agent_action_num
                        elif not self.is_sorted:
                            idx += self.agent_action_num
                    else:
                        if self.is_action_threshold and is_state_reach_maximum:
                            control.update({"fraction": 0.1}) if self.active_action[0] != -1 else None
                            self.graph.info_graph[device_name][link_name][
                                stream_name
                            ].update({"fraction": 0.1})
                            action_idx = np.zeros(len(action_idx))
                        else:
                            his_action = self.graph.info_graph[device_name][link_name][stream_name]["fraction"]
                            next_action = self.update_action_from_history(action[0], his_action, "fraction")
                            self.graph.info_graph[device_name][link_name][
                                stream_name
                            ].update({"fraction": next_action})
                            control.update({"fraction": next_action}) if self.active_action[0] != -1 else None      #TODO: Here might have bugs when no file stream exist
                if not is_rtt_active and not self.is_sorted:
                    idx += self.agent_action_num
                        ## End Action Load Section

        return control, action_idx

    def get_cost(self, fraction):
        cost = 10
        his_fraction = 0
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
                        # cost += 1
                        if "file" in stream["file_name"]:
                            his_fraction += self.graph.info_graph[device_name][link_name][stream_name]["fraction"]
                        if target_rtt != 0 and stream["rtt"] is not None:
                            # cost += stream["rtt"] * 1000 > target_rtt
                            _cost = max(stream["rtt"] * 1000 - target_rtt, 0)
                            cost += _cost if _cost < self._cost_threshold else self._cost_threshold
        cost -= his_fraction * 10
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

    graph,lists = tc.cw_training_case()
    # graph.info_graph["phone"]["wlan_phone_"]["6201@96"]["active"] = False
    graph.ADD_STREAM(
        lists[0],
        port_number=6200,
        file_name="file_75MB.npy",
        duration=[0, 50],
        thru=0,
        tos=96,
        name="File",
    )
    port_id = 6201
    for lnk in lists[0:2]: 
        graph.ADD_STREAM(
            lnk,
            port_number=port_id,
            file_name="voice_0.05MB.npy",
            duration=[0, 50],
            thru=0.05,
            tos=128,
            target_rtt=18,
            name="Proj",
        )
    # graph.show()
    wlanController = wlanDQNController(
        [i / 20 for i in range(1, 20, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [1, 3, 5, 7, 9, 11],  # AIFSN
        10000,
        graph,
        batch_size=32,
        is_CDQN = True,
        is_k_cost= 5
    )
    # print(wlanController.get_state())
    wlanController.init_action_guess()
    state = wlanController.get_state()
    print(state)
    action, _ = wlanController.action_to_control(wlanController.get_state())
    print("action",action)
    print(_)
    fraction = action["fraction"]
    cost = wlanController.get_cost(fraction)
    state_ = wlanController.get_state()
    action, _ = wlanController.action_to_control(wlanController.get_state())
    print("action",action)
    print(_)
    # print(cost)
    wlanController.store_transition(state, _, cost, state_)
    # print(state, _, cost, state_)
    abs_path = os.path.dirname(os.path.abspath(__file__))
    wlanController.store_memory(abs_path+"/saved_data/temp.npy")

