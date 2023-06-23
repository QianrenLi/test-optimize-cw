#!/usr/bin/env python3
import os
import sys
import ctypes
from transmission_graph import Graph
from tap import Connector
from tap_pipe import ipc_socket
import test_case as tc
from training.system2train import wlanDQNController

import threading
import json
import time
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import traceback

abs_path = os.path.dirname(os.path.abspath(__file__))
COLORS = (
    plt.rcParams["axes.prop_cycle"].by_key()["color"]
    + list(mcolors.BASE_COLORS.keys())
    + list(mcolors.CSS4_COLORS.keys())
)  # type: ignore
# ['SteelBlue', 'DarkOrange', 'ForestGreen', 'Crimson', 'MediumPurple', 'RosyBrown', 'Pink', 'Gray', 'Olive', 'Turquoise']
# ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

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


## ==========================test======================================== ##
heuristic_fraction = 0.3
## ==========================test======================================== ##


## ==================Config parameters================================== ##
_duration = 300
START_POINT = 0
control_period = 1
DURATION = int(_duration + START_POINT * control_period)
CONTROL_ON = True
control_times = (DURATION ) / control_period
experiment_name = "test"
dynamic_plotting = True
multiple_IC = True
is_Intel = False
is_local_test = False
init_thottle = 30
## ==================Config parameter================================== ##


## ==================threading parameter================================= ##
is_control = threading.Event()
is_collect = threading.Event()
is_draw = threading.Event()
is_writing = threading.Lock()
is_network_use = threading.Event()
is_stop = threading.Event()
is_control_stop = threading.Event()
return_num = threading.Semaphore(0)
## ==================threading parameter================================= ##

data_graph = {}
throttle = {}
system_return = {}
file_stream_nums = 0

name_dict = {"throughput": "thru"}
tos_to_ac = {"196": 0, "128": 1, "96": 2, "32": 3}

## =======================DQN Controller================================= ##
wlanController = None
cost_f = open(
    f"./training/logs/log-cost-%s.txt"
    % time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
    "a+",
)
## =======================DQN Controller================================= ##

temp_cw = {}
temp_idx = {}


## Result preprocessing component
def _sum_file_thru(outputs):
    """
    From the output get by applying connector build from _transmission_block, calculate the summation of throughput
    """
    thrus = 0
    try:
        outputs = [n for n in outputs if n]
        print(outputs)
        for output in outputs:
            output = eval(output["file_thru"])
            if type(output) == float:
                thrus += output
            else:
                thrus += float(output[0])
        return thrus
    except Exception as e:
        print(outputs)
    return 0


def _calc_rtt(graph):
    """
    Construct a rrt calculation ready connector waiting to be applied
    """
    conn = Connector()

    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            prot, sender, receiver = link_name.split("_")
            for stream_name, stream in streams.items():
                port, tos = stream_name.split("@")
                if stream["thru"] != 0:
                    conn.batch(sender, "read_rtt", {"port": port, "tos": tos}).wait(0.1)
    return conn.executor.wait(0.5)


def _throttle_calc(graph: Graph):
    global file_stream_nums, this_throttle_fraction
    # detect whether the num of file stream changes
    current_file_stream_nums = _update_file_stream_nums(graph)
    reset_flag = file_stream_nums == 0 and current_file_stream_nums != 0
    # this_throttle_fraction = _update_throttle_fraction(
    #     "one_dimensional_search", graph)
    this_throttle_fraction = heuristic_fraction
    # update throttle
    # print("this_throttle_fraction", this_throttle_fraction)
    if this_throttle_fraction:
        file_stream_nums = current_file_stream_nums
        port_throttle = graph.update_throttle(this_throttle_fraction, reset_flag)
    else:
        port_throttle = None
    return port_throttle


def _rtt_port_associate(graph, outputs):
    """
    Associate rtt value lists to the corresponding ports id (for better vision)
    """
    rtt_value = {}
    rtt_list = []
    idx = 0
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            prot, sender, receiver = link_name.split("_")
            for stream_name, stream in streams.items():
                port, tos = stream_name.split("@")
                if stream["thru"] != 0:
                    rtt_value.update({stream_name: float(outputs[idx]["rtt"])})
                    rtt_list.append(float(outputs[idx]["rtt"]))
                    idx += 1
    import numpy as np

    print(np.round(np.array(rtt_list) * 1000, 3))
    return rtt_value


## ip setup component
def _ip_extract(keyword: str, graph: Graph):
    """
    Extract ip from controlled device, store ip table into temp/ip_table.json
    """
    conn = Connector()
    ip_table = {}
    for device_name, links in graph.graph.items():
        if keyword is not None:
            conn.batch(device_name, "read_ip_addr", {"keyword": keyword})
        else:
            conn.batch(device_name, "read_ip_addr", {"keyword": "p2p\\|wlan"})
    outputs = conn.executor.wait(1).fetch().apply()
    results = [o["ip_addr"] for o in outputs]
    for r, c in zip(results, graph.graph.keys()):
        ip_table.update({c: {}})
        try:
            ipv4_addrs = eval(r)
        except:
            print("Error: client %s do not exist valid ipv4 addr" % c)
        else:
            # print(ipv4_addrs)
            # Generally multiple ipv4 with same name might be detect, but we only need the first one
            for ipv4_addr in ipv4_addrs:
                ip_table[c].update({ipv4_addr[0]: ipv4_addr[1]})

    # print(ip_table)
    # save the dict into json file
    with open("./temp/ip_table.json", "w") as f:
        json.dump(ip_table, f)


def _setup_ip(graph):
    """
    Set up ip stored in graph by the ip_table (requirement to setup the transmission)
    """
    if not multiple_IC:
        with open("./temp/ip_table.json", "r") as ip_file:
            ip_table = json.load(ip_file)

        for device_name in ip_table.keys():
            for protocol, ip in ip_table[device_name].items():
                graph.associate_ip(device_name, protocol, ip)
    else:
        _add_ind_according_to_interface(graph)
    # graph.show()


def _add_ind_according_to_interface(_graph:Graph):
    with open("./temp/ip_table.json", "r") as ip_file:
        ip_table = json.load(ip_file)
    ind_order = {}
    ip_val = {}
    for device_name in ip_table.keys():
        if is_Intel:
            ind = 0
        else:
            ind = 1
        for protocol, ip in ip_table[device_name].items():
            if is_Intel and "wlp" in protocol:
                ind_order.update({protocol: ind})
                ip_val.update({protocol : ip})
                ind += 1
            elif "wlx" in protocol:
                ind_order.update({protocol: ind})
                ip_val.update({protocol : ip})
                ind += 1
    print(ind_order)
    for device_name in ip_table.keys():    
        for link in _graph.graph[device_name].keys():
            prot, sender, receiver = link.split("_")
            _graph.info_graph[device_name][link].update({"ind": ind_order[prot]})
            _graph.associate_ip(device_name, prot, ip_val[prot])
## ipc communication component
def _add_ipc_port(graph):
    """
    Add ipc port (remote and local) to graph
    """
    port = 8888
    for device_name in graph.graph.keys():
        for link_name in graph.graph[device_name].keys():
            graph.info_graph[device_name][link_name].update({"ipc_port": port})
            graph.info_graph[device_name][link_name].update({"local_port": port - 1024})
            port += 1


def _loop_tx(sock: ipc_socket, *args):
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
            if is_stop.is_set():
                _buffer = None
                break
            _retry_idx += 1
            print("timeout\t", sock.link_name)
            continue
    return _buffer, _retry_idx


## transmission component
def _set_manifest(graph):
    """
    Setup config manifest required by tx to transmit
    """
    conn = Connector()
    # graph = Graph()
    # set manifest according to graph entries
    parameter_template = {
        "manifest_name": "manifest.json",
        "stream_idx": 0,
        "port": 0,
        "file_name": "",
        "tos": 100,
        "calc_rtt": False,
        "no_logging": True,
        "start": 0,
        "stop": 10,
        "throttle": 0,
    }
    for device_name, links in graph.graph.items():

        for link_name, streams in links.items():
            # init stream
            _init_parameters = []
            conn.batch(
                device_name,
                "init_stream",
                {"stream_num": len(streams), "manifest_name": link_name + ".json"},
            ).wait(0.5).apply()
            # add detail to manifest
            for port_number, stream in streams.items():
                parameter = parameter_template.copy()
                prot_tos = port_number.split("@")
                parameter.update({"manifest_name": link_name + ".json"})
                parameter.update({"port": int(prot_tos[0])})
                parameter.update({"tos": int(prot_tos[1])})
                parameter.update({"file_name": stream["file_name"]})
                if "file" not in stream["file_name"]:
                    parameter.update({"calc_rtt": True})
                else:
                    parameter.update({"throttle": init_thottle})
                parameter.update({"start": stream["duration"][0]})
                parameter.update({"stop": stream["duration"][1]})
                _init_parameters.append(parameter)
            # write detailed to device
            for i, _parameter in enumerate(_init_parameters):
                conn.batch(
                    device_name, "init_stream_para", {**_parameter, **{"stream_idx": i}}
                )
                print({**_parameter, **{"stream_idx": i}})
                conn.executor.wait(0.5)
            conn.executor.wait(0.5).apply()
    pass


def _transmission_block(graph):
    """
    Construct a transmission ready connector waiting to be applied
    """
    conn = Connector()
    # start reception
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            # split link name to protocol, sender, receiver
            prot, sender, receiver = link_name.split("_")
            receiver = receiver if receiver else ""
            for stream_name, stream in streams.items():
                # extract port number
                port_num, tos = stream_name.split("@")
                if "file" in stream["file_name"]:
                    conn.batch(
                        receiver,
                        "outputs_throughput",
                        {"port": port_num, "duration": DURATION},
                        timeout=DURATION + 5,
                    )
                else:
                    conn.batch(
                        receiver,
                        "outputs_throughput_jitter",
                        {
                            "port": port_num,
                            "duration": DURATION,
                            "calc_rtt": "--calc-rtt",
                            "tos": tos,
                        },
                        timeout=DURATION + 5,
                    )

    conn.executor.wait(1)
    # start transmission
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            # split link name to protocol, sender, receiver
            prot, sender, receiver = link_name.split("_")
            print("receiver", receiver)
            if receiver:
                ip_addr = graph.info_graph[receiver][prot + "_ip_addr"]
            else:
                ip_addr = "192.168.3.37" if not is_local_test else "127.0.0.1"
            conn.batch(
                sender,
                "run-replay-client",
                {
                    "target_addr": ip_addr,
                    "tx_addr": graph.info_graph[sender][prot + "_ip_addr"],
                    "duration": DURATION,
                    "manifest_name": link_name + ".json",
                    "ipc-port": graph.info_graph[sender][link_name]["ipc_port"],
                },
                timeout=DURATION + 5,
            )

    return conn.executor.wait(DURATION + 5)


def _loop_apply(conn):
    """
    Continuing apply the connector, fetch the result from remote until receiving outputs
    """
    conn.fetch()
    idx = 0
    while True:
        try:
            print("try to apply", idx)
            idx += 1
            outputs = conn.apply()
            return outputs
            break
        except Exception as e:
            print(e)
            break


## EDCA injection component
def _edca_default_params(graph: Graph, controls: dict):
    """
    Setup edca params prepared to be injected to remote
    """
    params = {}

    for link_name_tos in controls.keys():
        if "_" in link_name_tos:
            prot, tos, tx_device_name, rx_device_name = link_name_tos.split("_")
            if tx_device_name in graph.graph:
                if multiple_IC:
                    params[link_name_tos] = {
                        "ac": tos_to_ac[tos],
                        "cw_min": int(controls[link_name_tos]["cw"]),
                        "cw_max": int(controls[link_name_tos]["cw"]),
                        "aifs": int(controls[link_name_tos]["aifs"]),
                        "ind": graph.info_graph[tx_device_name][prot+"_"+tx_device_name+"_"+rx_device_name]["ind"],
                    }
                else:
                    is_realtek = "--realtek" if prot == "wlx" else ""
                    params[link_name_tos] = {
                        "ac": tos_to_ac[tos],
                        "cw_min": int(controls[link_name_tos]["cw"]),
                        "cw_max": int(controls[link_name_tos]["cw"]),
                        "aifs": int(controls[link_name_tos]["aifs"]),
                        "realtek": is_realtek,
                    }                    
    return params


def _set_edca_parameter(conn: Connector, params):
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


## Ongoing plot component
def _init_figure():
    lines = []
    axs = []
    fig = plt.figure(figsize=(12.8, 9.6))
    plt.ion()
    subplot_num = 2
    for _idx in range(subplot_num):
        _ax = fig.add_subplot(211 + _idx)
        # lines.append(_line)
        axs.append(_ax)
    return fig, axs


def _update_fig(fig, axs, data_graph):
    import numpy as np

    global START_POINT
    [ax.clear() for ax in axs]

    idx_to_key = ["rtts", "throttles"]
    idx_to_names = ["RTT (unit: ms)", "Fraction"]
    for _idx in range(len(axs)):
        x_axs = [0, 1]
        y_axs = [0, 1]
        legends = []
        colors_iter = iter(COLORS)
        for device_name, links in data_graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    if len(stream["indexes"]) > START_POINT:
                        if idx_to_key[_idx] in stream.keys():
                            if idx_to_key[_idx] == "throttles":
                                for _temp_key in temp_cw:
                                    for _temp_action in temp_cw[_temp_key]:
                                        c = next(colors_iter)
                                        vector_y = temp_cw[_temp_key][_temp_action]
                                        vector_x = (
                                            np.array(temp_idx[_temp_key][_temp_action])
                                        ) * control_period
                                        (_line,) = axs[_idx].plot(
                                            range(len(vector_x)), ".-", color=c
                                        )
                                        _line.set_xdata(vector_x)
                                        _line.set_ydata(vector_y)
                                        legends.append("%s-%s" % (_temp_key, _temp_action))

                            c = next(colors_iter)
                            (_line,) = axs[_idx].plot(
                                range(len(stream["indexes"])), ".-", color=c
                            )

                            vector_x = (
                                np.array(stream["indexes"][START_POINT:]) - START_POINT
                            ) * control_period

                            vector_y = stream[idx_to_key[_idx]][START_POINT:]
                            _line.set_xdata(vector_x)
                            _line.set_ydata(vector_y)
                            legends.append(stream_name)

                            x_axs[0] = min(x_axs[0], min(vector_x))
                            x_axs[1] = max(x_axs[1], max(vector_x))
                            y_axs[0] = min(y_axs[0], min(vector_y))
                            if idx_to_key[_idx] == "rtts":
                                y_axs[1] = min(max(y_axs[1], max(vector_y)), 60)
                            else:
                                y_axs[1] = max(y_axs[1], max(vector_y))

        axs[_idx].set_xlabel("time (s)")
        axs[_idx].set_ylabel(idx_to_names[_idx])
        axs[_idx].set_ylim(y_axs[0] * 0.9, y_axs[1] * 1.1)
        axs[_idx].set_xlim(x_axs[0] * 0.9, x_axs[1] * 1.1)
        axs[_idx].legend(legends)
        plt.show()

    fig.canvas.draw()
    fig.canvas.flush_events()


def _extract_data_from_graph(graph, data_graph, index):
    global throttle
    print("throttle", throttle)
    # Construct a temp graph
    for device_name, links in graph.graph.items():
        # update graph
        if device_name not in data_graph.keys():
            data_graph[device_name] = {}
        for link_name, streams in links.items():
            # update link
            if link_name not in data_graph[device_name].keys():
                data_graph[device_name][link_name] = {}

            # update throttle
            prot, sender, receiver = link_name.split("_")
            throttle_name = "throttle+" + sender + "+" + receiver
            if link_name in throttle.keys():
                # extract link name to devices
                if throttle_name not in data_graph[device_name][link_name].keys():
                    data_graph[device_name][link_name][throttle_name] = {
                        "indexes": [],
                        "throttles": [],
                    }

            for stream_name, stream in streams.items():
                # derive name from file name
                _stream_name = graph.info_graph[device_name][link_name][stream_name][
                    "name"
                ]
                # update stream
                if _stream_name not in data_graph[device_name][link_name].keys():
                    data_graph[device_name][link_name][_stream_name] = {
                        "indexes": [],
                        "rtts": [],
                        "thrus": [],
                    }
                # append rtt and throughput
                if (
                    graph.info_graph[device_name][link_name][stream_name]["active"]
                    == True
                ):
                    try:
                        if (
                            index
                            in data_graph[device_name][link_name][_stream_name][
                                "indexes"
                            ]
                        ):
                            data_graph[device_name][link_name][_stream_name]["thrus"][
                                -1
                            ] += stream["throughput"]
                        else:
                            data_graph[device_name][link_name][_stream_name][
                                "rtts"
                            ].append(stream["rtt"] * 1000)
                            data_graph[device_name][link_name][_stream_name][
                                "indexes"
                            ].append(index)
                            data_graph[device_name][link_name][_stream_name][
                                "thrus"
                            ].append(stream["throughput"])
                    except Exception as e:
                        print(e)
                        print("Data collect error", device_name, link_name, stream_name)
                    # ==============================================================================#
                    if link_name in throttle.keys() and "File" in _stream_name:
                        if (
                            index
                            in data_graph[device_name][link_name][throttle_name][
                                "indexes"
                            ]
                        ):
                            data_graph[device_name][link_name][throttle_name][
                                "throttles"
                            ][-1] += 0

                        else:
                            data_graph[device_name][link_name][throttle_name][
                                "indexes"
                            ].append(index)
                            data_graph[device_name][link_name][throttle_name][
                                "throttles"
                            ].append(this_throttle_fraction)
    pass


def _plot_exit():
    global data_graph, temp_cw, temp_idx
    data_graph = {}
    temp_cw = {}
    temp_idx = {}


## Control component
def _update_file_stream_nums(graph):
    """
    Inbuilt function to calculate active file stream in graph
    """
    file_stream_nums = 0
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            for stream_name, stream in streams.items():
                if (
                    "file" in stream["file_name"]
                    and graph.info_graph[device_name][link_name][stream_name]["active"]
                    == True
                ):
                    file_stream_nums += 1
    return file_stream_nums


def _update_throttle_fraction(algorithm_type, graph, **kwargs):
    # get target value from info graph
    target_rtt = 1000
    if algorithm_type == "one_dimensional_search":
        rtt_value = 0

        observed_rtt_list = list()
        target_rtt_list = list()

        # compute the maximum rtt of the graph
        for device_name, links in graph.graph.items():
            for link_name, streams in links.items():
                for stream_name, stream in streams.items():
                    # comparing requirements, skip file
                    try:
                        target_rtt = graph.info_graph[device_name][link_name][
                            stream_name
                        ]["target_rtt"]
                        if (
                            graph.info_graph[device_name][link_name][stream_name][
                                "active"
                            ]
                            == True
                            and target_rtt != 0
                        ):
                            rtt_value = graph.graph[device_name][link_name][
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


## Main threading section
def plot_thread():
    global data_graph
    fig, axs = _init_figure()
    index = 0
    while True:
        # wait until is draw
        is_draw.wait()
        if is_stop.is_set():
            break
        _update_fig(fig, axs, data_graph)
        index += 1
        is_draw.clear()
    # close the graph
    plt.title(experiment_name)
    fig.savefig("temp/%s-%s.png" % (experiment_name,time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    fig.clear()
    plt.ioff()
    plt.close(fig)


def transmission_thread(graph):
    global is_stop
    is_stop.clear()
    conn = _transmission_block(graph)
    results = _sum_file_thru(_loop_apply(conn))
    conn = _calc_rtt(graph)
    print(_rtt_port_associate(graph, _loop_apply(conn)))
    is_stop.set()
    return results

def DQN_training_thread():
    global wlanController
    loss_collect = 0
    counter = 0
    counter_max = 100
    while True:
        if is_stop.is_set():
            break
        is_network_use.wait()
        if wlanController.memory_counter > 4:
            loss = wlanController.training_network()
            loss_collect += loss
            counter += 1
            if counter > counter_max:
                print("loss\t", loss_collect / counter_max)
                loss_collect = 0
                counter = 0
            time.sleep(0.1)
        else:
            time.sleep(0.5)
            # print("training_thread",wlanController.memory_counter)

        # if wlanController.training_counter % 300 == 0:
        #     wlanController.store_params("%s/training/model/%s.pkl" % (abs_path, experiment_name))
        is_network_use.set()
    wlanController.store_params("%s/training/model/%s.pkl" % (abs_path, experiment_name))
    wlanController.store_memory(
        "%s/training/saved_data/%s-%d-%d"
        % (
            abs_path,
            experiment_name,
            wlanController.state_size,
            wlanController.action_size,
        )
        + "-"
        + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        + ".npy"
    )


def control_thread(graph, time_limit, period, socks):  # create a control_thread
    # start control and collect data
    global system_return, throttle, data_graph, wlanController, heuristic_fraction

    control_times = 0
    conn = Connector()

    state = None
    state_ = None
    his_fraction = heuristic_fraction
    is_network_use.set()
    while control_times < time_limit:
        ## collect data
        print("send statistics")
        for sock in socks:
            _buffer, _retry_idx = _loop_tx(sock, "statistics")
            if _buffer is None:
                break
            link_return = json.loads(str(_buffer.decode()))

            print("statistics return", _retry_idx, link_return)
            sys.stdout.flush()
            system_return.update({sock.link_name: link_return["body"]})
            time.sleep(0.1)

        ## Determine break -- Generally speaking, it is not a good choice to use thread unsafe parameter as a condition
        if is_stop.is_set():
            break
        ## update graph: activate and deactivate function
        graph.update_graph(system_return)
        ## DQN update and control
        try:
            is_network_use.wait()
            wlanController.update_graph(graph)
            if state is not None:
                state_ = wlanController.get_state()
            else:
                state = wlanController.get_state()

            if state_ is not None:
                cost = wlanController.get_cost(his_fraction)
                cost_f.write("{:.6f}\n".format(cost))
                wlanController.store_transition(state, action_idx, cost, state_)
                state = state_

            is_network_use.set()

            controls, action_idx = wlanController.action_to_control(state)
            if controls is None:
                continue
            print(controls)
            print("training_counter", wlanController.training_counter)
            if "fraction" in controls.keys():
                his_fraction = controls["fraction"]
                heuristic_fraction = his_fraction

        except Exception as e:
            traceback.print_exc()
            # print(e)
        ##
        if CONTROL_ON:
            print("send throttle")
            print("heuristic_fraction", heuristic_fraction)
            if port_throttle := _throttle_calc(graph):
                # print(port_throttle)
                throttle.update(port_throttle)

                # start control
                is_control.set()
                for sock in socks:
                    if sock.link_name in throttle.keys():
                        sock.ipc_transmit("throttle", throttle[sock.link_name])
                    else:
                        sock.ipc_transmit("throttle", {})
            else:
                for sock in socks:
                    sock.ipc_transmit("throttle", {})
                print("=" * 50)
                print("Control Stop")
                print("=" * 50)
            ## Get edca parameter
            edca_params = _edca_default_params(graph, controls)
            ## store cw value for plot
            for _device_name in controls:
                if _device_name != "fraction":
                    ## init temp_cw, temp_idx
                    if _device_name not in temp_cw:
                        temp_cw.update({_device_name: {}})
                        temp_idx.update({_device_name: {}})
                    for _action in controls[_device_name]:
                        if _action not in temp_cw[_device_name]:
                            temp_cw[_device_name].update({_action: []})
                            temp_idx[_device_name].update({_action: []})
                        maximum_action_value = 70 if _action == "cw" else 25
                        temp_cw[_device_name][_action].append(controls[_device_name][_action] / maximum_action_value)
                        temp_idx[_device_name][_action].append(control_times)

            ## Set edca parameter
            # _set_edca_parameter(conn, edca_params)

        ## plot data
        if dynamic_plotting:
            _extract_data_from_graph(graph, data_graph, control_times)
            is_draw.set()
        ## sleep
        control_times += 1
        time.sleep(period)
    ##
    print("main thread stopping")
    time.sleep(0.5)
    is_draw.set()
    ## close socket
    [sock.close() for sock in socks]
    cost_f.write("\n")
    is_control_stop.set()
    print("main thread stopped")

## Threading Entry
def start_testing_threading(graph, ctl_prot):
    # init_transmission thread
    tx_thread = threading.Thread(target=transmission_thread, args=(graph,))

    # init socket thread
    socks = []
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            # start threads to send data
            prot, sender, receiver = link_name.split("_")
            ip_addr = graph.info_graph[sender][prot + "_ip_addr"]               ## since no p2p/lo is considered
            sock = ipc_socket(
                ip_addr,
                graph.info_graph[device_name][link_name]["ipc_port"],
                local_port=graph.info_graph[device_name][link_name]["local_port"],
                link_name=link_name,
            )
            socks.append(sock)
        # init control thread
    control_t = threading.Thread(
        target=control_thread, args=(graph, control_times, control_period, socks)
    )
    training_t = threading.Thread(target=DQN_training_thread)
    tx_thread.start()
    training_t.start()
    time.sleep(0.5)
    control_t.start()
    # plot_thread()
    
## Throughput setup
def init_throughput_measure():
    print("Start Measure")
    global DURATION, init_thottle
    _saved_duration = DURATION
    _saved_throttle = init_thottle
    DURATION = 10
    init_thottle = 0
    port = 6200
    MCSs = []
    _graph, _lists = tc.cw_training_case()
    for _file_link in _lists:
        _graph, _ = tc.cw_training_case()
        port += 1
        _graph.ADD_STREAM(
            _file_link,
            port_number=port,
            file_name="file_75MB.npy",
            duration=[0, DURATION],
            thru=0,
            tos=96,
            name="File",
        )
        _setup_ip(_graph)
        _add_ipc_port(_graph)
        _set_manifest(_graph)
        _graph.show()
        results = transmission_thread(_graph)
        if results:
            print("results",results)
            MCSs.append(results)


    DURATION = _saved_duration
    init_thottle = _saved_throttle
    print("End of measure")
    return MCSs

## Iterative training over all the stream
def iter_all_training_scenario(ind, MCSs):
    from itertools import combinations
    global throttle
    _, _lists = tc.cw_training_case()
    # for _ind in range(len(_lists), ind - 1, -1):
    for _ind in range(ind,len(_lists)+ 1):
        for _file_link in _lists:
            for comb in combinations(_lists, _ind):
                _graph, _ = tc.cw_training_case(MCSs)
                _graph.ADD_STREAM(
                    _file_link,
                    port_number=6200,
                    file_name="file_75MB.npy",
                    duration=[0, DURATION],
                    thru=0,
                    tos=96,
                    name="File",
                )
                port_id = 6201
                for _link in comb:
                    _graph.ADD_STREAM(
                        _link,
                        port_number=port_id,
                        file_name="proj_6.25MB.npy",
                        duration=[0, DURATION],
                        thru = 7 * 8,
                        tos= 128,
                        target_rtt=18,
                        name="Proj-%s" % _link,
                    )
                    port_id += 1
                _graph.show()
                try:
                    print("============= Start one test period ==================")
                    is_stop.clear()
                    is_control_stop.clear()
                    throttle = {}
                    _setup_ip(_graph)
                    _add_ipc_port(_graph)
                    _set_manifest(_graph)
                    _graph.show()
                    start_testing_threading(_graph, "wlp")
                    plot_thread()
                    is_stop.wait()                                      # wait until all thread stop
                    is_control_stop.wait()
                    _plot_exit()
                    print("============= Stop one test period =============")
                except Exception as e:
                    print("===== %d-th transmission Error ====="%_ind, e)
                    break

    cost_f.close()


## local training test case
def local_training():
    graph,_ = tc.cw_training_case()
    wlanController = wlanDQNController(
        [i / 10 for i in range(1, 10, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [1, 5 , 10, 15, 20],  # AIFSN
        10000,
        graph,
        batch_size= 128,
        gamma = 0.9,
        is_CDQN = True,
        is_k_cost= 5
    )
    memory_path_first_3 = "6-20-2-8-63-2023-06-20-17:57:35.npy" # loss -> 0.3
    memory_path_first_4 = "6-20-2-8-63-2023-06-20-18:01:05.npy" # loss -> 1.1
    memory_path_first_5 = "6-20-2-8-63-2023-06-20-18:04:37.npy" # loss -> 1.7
    memory_path_all = "6-20-2-8-63-2023-06-20-18:08:09.npy"     # loss -> 2
    # wlanController.load_memory(abs_path + "/training/saved_data/%s" % memory_path_first_5)
    wlanController.load_memory_test(abs_path + "/training/saved_data/%s" % memory_path_first_3, abs_path + "/training/saved_data/%s" % memory_path_all) # loss -> 7
    # wlanController.load_memory(abs_path + "/training/saved_data/test-8-53-2023-06-18-15:56:04.npy")
    loss_saved = 0
    episode = 1000
    num_episode = 10
    counter_max = 50
    # while True:
    for i_episode in range(num_episode):
        counter = 0
        for i in range(episode):
            loss = wlanController.training_network()  
            loss_saved += loss 
            if counter % counter_max == 0:
                print("loss\t", loss_saved/counter_max) 
                sys.stdout.flush()
                loss_saved = 0
            counter += 1
    wlanController.store_params("%s/training/model/%s.pkl" % (abs_path, "test2"))
            
        
         
def main(args):
    global experiment_name, wlanController
    experiment_name = args.experiment_name
    # local_training()
    # exit()
    graph,_ = tc.cw_training_case()
    # graph = tc.cw_test_case(DURATION)
 
    # wlan - termux wifi ip  ; p2p - Wifi Direct ip
    # wlp - Intel IC wifi ip ; wlx - Realtek IC (rtl8812au) wifi ip
    _ip_extract("wlan\\|p2p\\|wlx\\|wlp", graph)

    MCSs = [400, 400]
    MCSs = init_throughput_measure()
    # exit()
    # graph.show()
    # exit()
    # _setup_ip(graph)
    # graph.associate_ip("TV", "wlx", "192.168.3.61")
    # _add_ipc_port(graph)
    # exit()
    wlanController = wlanDQNController(
        [i / 20 for i in range(1, 20, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [1, 5, 10, 15, 20],  # AIFSN
        10000,
        graph,
        batch_size=32,
        is_CDQN = False,
        is_k_cost= 5
    )
    if args.load:
        wlanController.load_params("%s/training/model/%s.pkl" % (abs_path, experiment_name))
        wlanController.load_memory(abs_path + "/training/saved_data/6-20-3-8-63-2023-06-20-20:58:13.npy")
        wlanController.action_counter = 1000
    
    # exit()
    # _set_manifest(graph)
    # graph.show()
    # exit()
    iter_all_training_scenario(1, MCSs)
    # start_testing_threading(graph, "wlp")
    # TODO: fix dynamic plot issue
    # if dynamic_plotting:
    #     plot_thread()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--scenario", type=int, default=1, help="scenario in 1,2,3"
    )
    parser.add_argument(
        "-n", "--experiment_name", type=str, default="test", help="experiment name"
    )
    parser.add_argument("--load", action="store_true", help="load model")
    args = parser.parse_args()
    main(args)
