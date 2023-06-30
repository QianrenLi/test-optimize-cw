import os
import json
import sys
import ctypes
import numpy as np
import threading
import time

abs_path = os.path.dirname(os.path.abspath(__file__))

current = abs_path
parent = os.path.dirname(current)
sys.path.append(parent)
from transmission_graph import Graph
from tap import Connector
from tap_pipe import ipc_socket

## ====================================================================== ##
os.system("make")
NATIVE_MOD = ctypes.CDLL("./liboptimize.so")
NATIVE_MOD.update_throttle_fraction.restype = ctypes.c_float
NATIVE_MOD.update_throttle_fraction.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
NATIVE_MOD.init_throttle_fraction.restype = ctypes.c_float
NATIVE_MOD.init_throttle_fraction.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
def _list_to_c_array(arr: list, arr_type=ctypes.c_float):
    return (arr_type * len(arr))(*arr)
## ====================================================================== ##



class txController():
    '''
    This class is used to control the transmission of the network

    Func:
        Communicate with ongoing tx users (tx object from txAgent)
    '''
    def __init__(self, graph : Graph, is_stop: threading.Event, socks:list, action_space:list, control_interval = 1) -> None:
        """
        Initial controller with graph, which provide the communication necessary information, is_stop event which used to stop the controlling,
        and ipc_sockets used to communicate with ongoing device
        Args:
            graph (Graph): graph object
            is_stop (threading.Event): stop event
            socks (list): list of ipc_socket
            control_interval (int, optional): control interval. Defaults to 1 (s).
        """
        self.graph = graph
        self.is_stop = is_stop
        self.socks = socks
        self.action_space = action_space
        self.control_interval = control_interval


        self.is_line_search_policy = False                          # With enable the Line control algorithm, 
                                                                                    # the throttle will compute according to the line search algorithm
        self.is_local_test = False
        self.CONTROL_ON = True

        self.tos_to_ac = {"196": 0, "128": 1, "96": 2, "32": 3}
        self.throttle = {}
        self.system_return = {}
        self.heuristic_fraction = 0
        self.his_file_num = 0




    def _loop_tx(self, sock: ipc_socket, *args):
        """
        Continuous transmitting to the remote ipc socket until the transmission is successful
        """
        _retry_idx = 0
        print("Collect\t", sock.link_name)
        while True:
            try:
                _buffer = sock.ipc_communicate(*args)
                break
            except Exception as e:
                print(e)
                if self.is_stop.is_set():
                    _buffer = None
                    break
                _retry_idx += 1
                print("timeout\t", sock.link_name)
                continue
        return _buffer, _retry_idx

    def _update_throttle_fraction(self, algorithm_type):
        """
        Update the throttle fraction of the graph
        """
        # get target value from info graph
        target_rtt = 1000
        if algorithm_type == "one_dimensional_search":
            rtt_value = 0

            observed_rtt_list = list()
            target_rtt_list = list()

            # compute the maximum rtt of the graph
            for device_name, links in self.graph.graph.items():
                for link_name, streams in links.items():
                    for stream_name, stream in streams.items():
                        # comparing requirements, skip file
                        try:
                            target_rtt = self.graph.info_graph[device_name][link_name][
                                stream_name
                            ]["target_rtt"]
                            if (
                                self.graph.info_graph[device_name][link_name][stream_name][
                                    "active"
                                ]
                                == True
                                and target_rtt != 0
                            ):
                                rtt_value = self.graph.graph[device_name][link_name][
                                    stream_name
                                ]["rtt"]
                                observed_rtt_list.append(rtt_value * 1e3)
                                target_rtt_list.append(target_rtt)
                        except:
                            continue
            print(observed_rtt_list)
            print(target_rtt_list)
            length = len(observed_rtt_list)
            observed_rtt_list = _list_to_c_array(observed_rtt_list)
            target_rtt_list = _list_to_c_array(target_rtt_list)
            this_throttle_fraction = NATIVE_MOD.update_throttle_fraction(
                length, observed_rtt_list, target_rtt_list
            )
            return this_throttle_fraction
        return 0.1

    ## Control component
    def _update_file_stream_nums(self):
        """
        Inbuilt function to calculate active file stream in graph
        """
        file_stream_nums = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if (
                        "file" in stream["file_name"]
                        and self.graph.info_graph[device_name][link_name][stream_name]["active"]
                        == True
                    ):
                        file_stream_nums += 1
        return file_stream_nums

    def _throttle_calc(self):
        """
        Compute exact throttle (Throughput) of each file transmission
        """
        # detect whether the num of file stream changes
        file_stream_nums = self._update_file_stream_nums()
        reset_flag = self.his_file_num == 0 and file_stream_nums != 0
        if self.is_line_search_policy:
            this_throttle_fraction = self._update_throttle_fraction(
                "one_dimensional_search")
        else:
            this_throttle_fraction = self.heuristic_fraction

        if this_throttle_fraction:
            self.his_file_num = file_stream_nums
            port_throttle = self.graph.update_throttle(this_throttle_fraction, reset_flag)
        else:
            port_throttle = None
        return port_throttle


    ## EDCA injection component
    def _edca_default_params(self, controls: dict):
        """
        Setup edca params prepared to be injected to remote
        """
        params = {}

        for link_name_tos in controls.keys():
            if "_" in link_name_tos:
                prot, tos, tx_device_name, rx_device_name = link_name_tos.split("_")
                if tx_device_name in self.graph.graph:
                    params[link_name_tos] = {
                        "ac": self.tos_to_ac[tos],
                        "cw_min": int(controls[link_name_tos]["cw"]),
                        "cw_max": int(controls[link_name_tos]["cw"]),
                        "aifs": int(controls[link_name_tos]["aifs"]),
                        "ind": self.graph.info_graph[tx_device_name]["ind"],
                    }
                
        return params


    def _set_edca_parameter(self, conn: Connector, params):
        """
        Inject EDCA parameter change to remote
        """
        for link_name_tos in params:
            device_name = link_name_tos.split("_")[2]
            conn.batch(device_name, "modify_edca", params[link_name_tos])
            print(params[link_name_tos])
            conn.executor.wait(0.1)
        conn.executor.wait(0.1).apply()
        return conn


    def _generate_controls(self, index: int): 
        """
        Generate controls for each iteration (Collect data only)
        1. EDCA Parameter
        2. Throttle
        """
        control = {}
        fraction = 0
        print("Control index", index)

        fraction_num = len(self.action_space[0])
        cw_aifs_num = [len(self.action_space[1]),  len(self.action_space[2])]


        fraction_counter = index % fraction_num
        index = int(index / fraction_num)

        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                prot, sender, receiver = link_name.split("_")
                for stream_name, stream in streams.items():
                    port, tos = stream_name.split("@")
                    if "file" not in stream["file_name"]:
                        _control = {}
                        for _idx, action_name in enumerate(["cw", "aifs"]):
                            _control.update({action_name: self.action_space[_idx + 1][index % cw_aifs_num[_idx]] })
                            index = int(index / cw_aifs_num[_idx])
                        control.update(
                            {prot + "_" + tos + "_" + sender + "_" + receiver: _control}
                        )
                    else:
                        control.update(
                            {"fraction": self.action_space[0][fraction_counter]}
                        )
        return control


    def control_thread(self):  # create a control_thread
        """
        Control thread:

        The control thread is responsible for collecting data and controlling the system, 
        default the control is implemented by _generate_control function.
        It is recommended to be reconstructed by the user to implement the control algorithm.
        """
        # start control and collect data
        system_return = {}
        throttle = {}

        control_times = 0
        conn = Connector()

        while True:
            control_times += 1
            ## collect data
            print("send statistics")
            for sock in self.socks:
                _buffer, _retry_idx = self._loop_tx(sock, "statistics")
                if _buffer is None:
                    break
                link_return = json.loads(str(_buffer.decode()))

                print("statistics return", _retry_idx, link_return)
                sys.stdout.flush()
                system_return.update({sock.link_name: link_return["body"]})
                time.sleep(0.1)

            ## Determine break -- Generally speaking, it is not a good choice to use thread unsafe parameter as a condition
            if self.is_stop.is_set():
                break

            ## update graph: activate and deactivate function
            self.graph.update_graph(system_return)
            ## Get edca parameter
            controls = self._generate_controls(control_times)

            if self.CONTROL_ON:
                if port_throttle := self._throttle_calc():
                    # print(port_throttle)
                    throttle.update(port_throttle)

                    for sock in self.socks:
                        if sock.link_name in throttle.keys():
                            sock.ipc_transmit("throttle", throttle[sock.link_name])
                        else:
                            sock.ipc_transmit("throttle", {})
                else:
                    for sock in self.socks:
                        sock.ipc_transmit("throttle", {})
                    print("=" * 50)
                    print("Control Stop")
                    print("=" * 50)


                edca_params = self._edca_default_params(controls)
                ## Set edca parameter
                self._set_edca_parameter(conn, edca_params)

            time.sleep(self.control_interval)
        
        ## close sockets
        [sock.close() for sock in self.socks]