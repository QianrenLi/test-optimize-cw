import os
import json
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from transmission_graph import Graph

class envCap:
    def __init__(self, graph:Graph, cost_func = None) -> None:
        self.graph = graph
        self.action_keys = ["cw", "aifs", "txop", "fraction"]
        self.action_key_zipper = {"cw": 0, "aifs": 0, "txop": 0, "fraction": 0}
        self.action_format = {"cw-aifs-txop-fraction":{}}
        self.state_format = {"rtt":{}}
        self.env_graph = {}
        self.cost_func = cost_func
        pass

    def active_stream_num(self):
        num = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if (
                        self.graph.info_graph[device_name][link_name][stream_name][
                            "active"
                        ]
                        == True
                    ):
                        num += 1
        return num

    def zip_action_value(self, action_key_zipper):
        return "%d-%d-%d-%d" % (action_key_zipper["cw"], action_key_zipper["aifs"], action_key_zipper["txop"], action_key_zipper["fraction"])

    ## Formulate the environment graph
    def system_formation(self):
        active_stream_num = self.active_stream_num()
        if active_stream_num not in self.env_graph:
            self.env_graph.update({active_stream_num:{}})
            for device_name, links in self.graph.graph.items():
                for link_name, streams in links.items():
                    for stream_name, stream in streams.items():
                        self.env_graph[active_stream_num].update({stream_name: self.action_format.copy()})
        return self


    ## Collect data to environment graph
    def collect_data(self):
        active_stream_num = self.active_stream_num()
        self.system_formation()
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    _state_format = self.state_format.copy()
                    # update the state keys (rtt)
                    for _key in _state_format:
                        if _key in self.graph.info_graph[device_name][link_name][stream_name]:
                            _state_format.update({_key: self.graph.info_graph[device_name][link_name][stream_name][_key]})
                    # update the action keys (cw, aifs, txop, fraction)
                    temp_zipper = self.action_key_zipper.copy()
                    for key in self.action_keys:
                        if key in self.graph.info_graph[device_name][link_name][stream_name]:
                            temp_zipper.update({key: self.graph.info_graph[device_name][link_name][stream_name][key]})
                    zip_key = self.zip_action_value(temp_zipper)
                    self.env_graph[active_stream_num][stream_name]["cw-aifs-txop-fraction"].update({zip_key: _state_format})
        return self


    def load_data(self, path:str):
        with open(path, "r") as f:
            self.env_graph = json.load(f)
        return self


    ## Save data to file
    def save_data(self, path:str):
        with open(path, "w") as f:
            json.dump(self.env_graph, f, indent=4)
        return self
    
    def get_state(self,active_stream_num:int, stream_name:str, action:str):
        if action in self.env_graph[active_stream_num][stream_name]["cw-aifs-txop-fraction"]:
            return self.env_graph[stream_name]["cw-aifs-txop-fraction"][action]["rtt"]
        else:
            raise Exception("Cost not found")


def test_env():
    import test_case as tc
    data_path = abs_path + "/env/env.json"
    graph,lists = tc.cw_training_case()
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
    envCapIns = envCap(graph)
    envCapIns.system_formation().collect_data()

test_env()