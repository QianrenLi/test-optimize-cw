#!/usr/bin/env python3
import os
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
heuristic_fraction = 0.1
## ==========================test======================================== ##


## ==================Constant parameter================================== ##
_duration = 600
START_POINT = 0
control_period = 0.8
DURATION = int(_duration * 1.7 + START_POINT * control_period)
rx_DURATION = int(DURATION)
CONTROL_ON = True
control_times = (DURATION - _duration * 0.7) / control_period
experiment_name = "test"
## ==================Constant parameter================================== ##


## ==================threading parameter================================= ##
is_control = threading.Event()
is_collect = threading.Event()
is_draw = threading.Event()
is_writing = threading.Lock()
is_network_use = threading.Event()
is_stop = False
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
cost_f = open(f"./training/logs/log-cost-%s.txt" % time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()
        ), "a+")
## =======================DQN Controller================================= ##
##

temp_cw = []

def _ip_extract(keyword, graph):
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
            for ipv4_addr in ipv4_addrs[::-1]:
                ip_table[c].update({ipv4_addr[0]: ipv4_addr[1]})

    # print(ip_table)
    # save the dict into json file
    with open("./temp/ip_table.json", "w") as f:
        json.dump(ip_table, f)


def _update_file_stream_nums(graph):
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


def _loop_tx(sock, *args):
    _retry_idx = 0
    while True:
        try:
            _buffer = sock.ipc_communicate(*args)
            break
        except Exception as e:
            print(e)
            _retry_idx += 1
            continue
    return _buffer, _retry_idx


def _blocking_wait(return_num, graph):
    return_num.release()
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            return_num._value -= 1
    return_num.acquire()


def _transmission_block(graph):
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
                        {"port": port_num, "duration": rx_DURATION},
                        timeout=DURATION + 5,
                    )
                else:
                    conn.batch(
                        receiver,
                        "outputs_throughput_jitter",
                        {
                            "port": port_num,
                            "duration": rx_DURATION,
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
                ip_addr = "192.168.3.37"
            conn.batch(
                sender,
                "run-replay-client",
                {
                    "target_addr": ip_addr,
                    "duration": DURATION,
                    "manifest_name": link_name + ".json",
                    "ipc-port": graph.info_graph[sender][link_name]["ipc_port"],
                },
                timeout=DURATION + 5,
            )

    return conn.executor.wait(DURATION + 5)


def _calc_rtt(graph):
    conn = Connector()

    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            prot, sender, receiver = link_name.split("_")
            for stream_name, stream in streams.items():
                port, tos = stream_name.split("@")
                if stream["thru"] != 0:
                    conn.batch(sender, "read_rtt", {"port": port, "tos": tos}).wait(0.1)
    return conn.executor.wait(0.5)


def _loop_apply(conn):
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


def _sum_file_thru(outputs):
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


def _rtt_port_associate(graph, outputs):
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


def _add_ipc_port(graph):
    port = 11100
    for device_name in graph.graph.keys():
        for link_name in graph.graph[device_name].keys():
            graph.info_graph[device_name][link_name].update({"ipc_port": port})
            graph.info_graph[device_name][link_name].update({"local_port": port - 1024})
            port += 1


def _name_tunnel(input_string):
    # using lambda and regex functions to tune the name
    return re.compile("|".join(name_dict.keys())).sub(
        lambda ele: name_dict[re.escape(ele.group(0))], input_string
    )


def _get_graph(scenario, DURATION):
    if scenario == 1:
        return tc.get_scenario_1_graph(DURATION)
    elif scenario == 2:
        return tc.get_scenario_2_graph(DURATION)
    elif scenario == 3:
        return tc.get_scenario_3_graph(DURATION)
    else:
        return tc.get_scenario_local_test(DURATION)


def _edca_default_params(graph: Graph, controls: dict):
    params = {}

    for device_name_tos in controls.keys():
        if "@" in device_name_tos:
            print(device_name_tos)
            device_name, tos = device_name_tos.split("@")
            if device_name in graph.graph:
                params[device_name] = {
                    "ac": tos_to_ac[tos],
                    "cw_min": int(controls[device_name_tos]),
                    "cw_max": int(controls[device_name_tos]),
                    "aifs": -1,
                }
    return params


def _set_edca_parameter(conn: Connector, params):
    for device_name in params:
        conn.batch(device_name, "modify_edca", params[device_name])
        conn.executor.wait(0.01)
    conn.executor.wait(0.01).apply()
    return conn


def _set_manifest(graph):
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
                    parameter.update({"no_logging": False})
                else:
                    parameter.update({"throttle": 30})
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


# function to set up ip address for each device


def _setup_ip(graph):
    with open("./temp/ip_table.json", "r") as ip_file:
        ip_table = json.load(ip_file)

    for device_name in ip_table.keys():
        for protocol, ip in ip_table[device_name].items():
            graph.associate_ip(device_name, protocol, ip)


# function for picture updating


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
                    c = next(colors_iter)
                    if len(stream["indexes"]) > START_POINT:
                        if idx_to_key[_idx] in stream.keys():

                            (_line,) = axs[_idx].plot(
                                range(len(stream["indexes"])), ".-", color=c
                            )
                            if idx_to_key[_idx] == "throttles":
                                vector_y = temp_cw
                                vector_x = list(range(len(vector_y)))
                                _line.set_xdata(vector_x)
                                _line.set_ydata(vector_y)
                            
                            (_line,) = axs[_idx].plot(
                                range(len(stream["indexes"])), ".-", color=c
                            )
                            
                            vector_x = (
                                np.array(stream["indexes"][START_POINT:]) - START_POINT
                            ) * control_period

                            vector_y = stream[idx_to_key[_idx]][START_POINT:]

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


def graph_plot():
    global data_graph
    fig, axs = _init_figure()
    index = 0
    while True:
        # wait until is draw
        is_draw.wait()
        if is_stop:
            break
        _update_fig(fig, axs, data_graph)
        index += 1
        is_draw.clear()
    # close the graph
    plt.title(experiment_name)
    fig.savefig("temp/%s.png" % experiment_name)
    fig.clear()
    plt.ioff()
    plt.close(fig)


# create a sub-threading to send data given an ipc socket


def transmission_thread(graph):
    conn = _transmission_block(graph)
    print(_sum_file_thru(_loop_apply(conn)))
    conn = _calc_rtt(graph)
    print(_rtt_port_associate(graph, _loop_apply(conn)))


def DQN_training_thread():
    import torch

    global is_stop, wlanController
    while True:
        if is_stop:
            break
        is_network_use.wait()
        if wlanController.memory_counter > 4:
            wlanController.training_network()
            time.sleep(0.01)
        else:
            time.sleep(0.5)
            # print("training_thread",wlanController.memory_counter)

        if wlanController.training_counter % 300 == 0:
            torch.save(
                wlanController.eval_net.state_dict(), "./temp/%s.pkl" % experiment_name
            )
        is_network_use.set()


def control_thread(graph, time_limit, period, socks):  # create a control_thread
    # start control and collect data
    global is_stop, system_return, throttle, data_graph, wlanController, heuristic_fraction

    control_times = 0
    conn = Connector()

    state = None
    state_ = None
    his_fraction = heuristic_fraction
    is_network_use.set()
    while control_times < time_limit:
        ## collect data
        for sock in socks:
            _buffer, _retry_idx = _loop_tx(sock, "statistics")
            link_return = json.loads(str(_buffer.decode()))

            print("statistics return", _retry_idx, link_return)
            system_return.update({sock.link_name: link_return["body"]})

        ## update graph
        graph.update_graph(system_return)
        # graph.show()
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
            print(controls)
            print("training_counter", wlanController.training_counter)
            if "fraction" in controls.keys():
                his_fraction = controls["fraction"]
                # his_fraction = 0.5
                heuristic_fraction = his_fraction
        except Exception as e:
            print(e)
        ##
        if CONTROL_ON:
            print("heuristic_fraction", heuristic_fraction)
            if port_throttle := _throttle_calc(graph):
                # print(port_throttle)
                throttle.update(port_throttle)

                # start control
                is_control.set()
                for sock in socks:
                    if sock.link_name in throttle.keys():
                        _buffer, _retry_idx = _loop_tx(
                            sock, "throttle", throttle[sock.link_name]
                        )
                    else:
                        _buffer, _retry_idx = _loop_tx(sock, "throttle", {})
            else:
                for sock in socks:
                    _buffer, _retry_idx = _loop_tx(sock, "throttle", {})
                print("=" * 50)
                print("Control Stop")
                print("=" * 50)
            ## set edca parameter
            edca_params = _edca_default_params(graph, controls)
            temp_cw.append(controls["PC@96"])
            _set_edca_parameter(conn, edca_params)
        ## plot data
        _extract_data_from_graph(graph, data_graph, control_times)
        is_draw.set()
        ## sleep
        control_times += 1
        time.sleep(period)
    ##
    print("main thread stopping")
    is_stop = True
    time.sleep(0.5)
    is_draw.set()
    print("main thread stopped")
    cost_f.close()


def start_testing_threading(graph, ctl_prot):
    # init_transmission thread
    tx_thread = threading.Thread(target=transmission_thread, args=(graph,))

    # init socket thread
    socks = []
    for device_name, links in graph.graph.items():
        for link_name, streams in links.items():
            # start threads to send data
            prot, sender, receiver = link_name.split("_")
            ip_addr = graph.info_graph[sender][ctl_prot + "_ip_addr"]
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


def main(args):
    global experiment_name, wlanController
    experiment_name = args.experiment_name
    graph = tc.cw_test_case(DURATION)
    if args.scenario > 0:
        _ip_extract("wlan\\|p2p\\|wlp", graph)
        _setup_ip(graph)
    _add_ipc_port(graph)
    graph.show()
    wlanController = wlanDQNController(
        [i / 10 for i in range(1, 10, 1)], [1, 10, 15, 25, 30, 35, 40, 45], 300, graph
    )
    # exit()
    _set_manifest(graph)
    if args.scenario > 0:
        start_testing_threading(graph, "wlp")
    else:
        start_testing_threading(graph, "lo")
    # transmission_thread(graph)

    # push matlab plot to main thread
    graph_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--scenario", type=int, default=1, help="scenario in 1,2,3"
    )
    parser.add_argument(
        "-n", "--experiment_name", type=str, default="test", help="experiment name"
    )
    args = parser.parse_args()
    main(args)
