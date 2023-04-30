#!/usr/bin/env python3
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from tap import Connector
from tap import TimeoutException
import ast
import sys
import argparse
import traceback


def ORDERED_CONNECTOR(): return [Connector(key) for key in CONFIG]


def CALC_THROUGHPUT(): return [sum([float(QOS[_stream][key])
                                    for key in QOS[_stream] if key.endswith('thru')]) for _stream in QOS]


# QOS parameters
REAL_LEN_WEIGHT = 20
PROJ_LEN_WEIGHT = 5
THRU_WEIGHT = 0.5

# Simulation parameters
DURATION = 1
global PORTS_NUM
PORTS_NUM = 6  # 7
MONITOR_IP = "127.0.0.1"

f = open(f'./logs/optimize_log_{time.ctime()}.json', 'a+')

CONFIG = {}
CONFIG_FILE = {}
CONFIG_QOE = {}
PORTS_NUM = 4


def optimize_scenario(scenario):
    global CONFIG, CONFIG_FILE, CONFIG_QOE, PORTS_NUM, CONTROL
    # Test Case I
    if scenario == 1:
        PORTS_NUM = 6
        CONFIG = {
            'phone':   {'p2p_PC.json': {'file_port': list(range(5201, 5201+PORTS_NUM)), 'proj_port': [6201], 'target_addr':  "", 'receiver': "PC"},
                        'wlan_PC.json': {'real_port': [6202], 'target_addr':  "", 'receiver': "PC"}},
            'PC':   {'wlan_phone.json': {'real_port': [6203], 'target_addr': "", 'receiver': "phone"}}
        }
        CONFIG_FILE = {
            'phone':  {'p2p_PC.json': {'file_port': "file_75MB.npy", "proj_port": "proj_6.25MB.npy", "real_port": "voice_0.05MB.npy"},
                       'wlan_PC.json': {'file_port': "file_75MB.npy", "proj_port": "proj_6.25MB.npy", "real_port": "voice_0.05MB.npy"}},
            'PC': {'wlan_phone.json': {'file_port': "file_75MB.npy", "proj_port": "proj_6.25MB.npy", "real_port": "voice_0.05MB.npy"}},
        }
        CONFIG_QOE = {
            "--calc-jitter": [6201, 6202, 6203],
            "--calc-rtt": [6201, 6202, 6203],
        }
        CONTROL = {
            'phone':   {'file_throttle': 0},
            'PC':   {'file_throttle': 0}
        }
    # Test Case II
    elif scenario == 2:
        PORTS_NUM = 3
        CONFIG = {
            'phone':   {'p2p_PC.json': {'file_port': list(range(5201, 5201+PORTS_NUM)), 'target_addr': '192.168.49.1', 'receiver': "PC"},
                        'wlan_PC_vo.json': {'real_port': [6201], 'target_addr':  '192.168.3.49', 'receiver': "PC"}
                        },
            'PC':   {'wlan_phone_vo.json': {'real_port': [6202], 'target_addr': '192.168.3.49', 'receiver': "phone"},  # 192.168.3.15
                     'p2p_phone.json': {'real_port': [6203], 'target_addr': "192.168.49.102", 'receiver': "phone"},
                     'p2p_pad.json': {'real_port': [6204], 'proj_port': [6205], 'file_port': list(range(5201 + PORTS_NUM + 6, 5201+2*PORTS_NUM+6)), 'target_addr': "192.168.49.28", 'receiver': "pad"}
                     },
            "pad": {},
        }

        CONFIG_FILE = {
            'phone':  {'p2p_PC.json': {'file_port': "file_75MB.npy"},
                       'wlan_PC_vo.json': {"real_port": "voice_0.05MB.npy"}
                       },
            'PC': {'wlan_phone_vo.json': {"real_port": "voice_0.05MB.npy"},
                   'p2p_phone.json': {"real_port": "kb_0.125MB.npy", 'file_port': "file_75MB.npy"},
                   'p2p_pad.json': {'file_port': "file_75MB.npy", "proj_port": "proj_6.25MB.npy", "real_port": "kb_0.125MB.npy"}
                   },
            'pad': {}
        }
        CONFIG_QOE = {
            "--calc-jitter": list(range(6201, 6206)),
            "--calc-rtt": list(range(6201, 6206)),
        }
        CONTROL = {
            'phone':   {'file_throttle': 0},
            'PC':   {'file_throttle': 0},
            'pad':   {'file_throttle': 0},
        }
    # Test Case III
    elif scenario == 3:
        PORTS_NUM = 3
        CONFIG = {
            'phone':
                {
                    'p2p_PC.json': {'file_port': list(range(5201,5201+PORTS_NUM)), 'proj_port': [6201], 'target_addr':  "", 'receiver': "PC"},
                    # 'wlan_PC.json': {'real_port': [6202], 'target_addr':  "", 'receiver': "PC"}
                },
            'PC':
                {
                    # 192.168.3.15
                    # 'wlan_phone_vo.json': {'real_port': [6203], 'target_addr': '192.168.3.49', 'receiver': "phone"},
                },
            "pad": {
                'p2p_TV.json': {'file_port': list(range(5201+PORTS_NUM,5201+PORTS_NUM*2)),'real_port': [6204], 'proj_port': [6205], 'target_addr':  "", 'receiver': "TV"}
            },
            "TV": {
                'p2p_pad.json': {'real_port': [6206], 'proj_port': [6207], 'target_addr':  "", 'receiver': "pad"}
            },
        }

        CONFIG_FILE = {
            'phone':  {'p2p_PC.json': {'file_port': "file_75MB.npy", 'proj_port': "proj_6.25MB.npy"},
                       'wlan_PC.json': {"real_port": "voice_0.05MB.npy"}},
            'PC': {'wlan_phone_vo.json': {"real_port": "voice_0.05MB.npy"}
                   },
            'pad': {
                'p2p_TV.json': {'file_port': "file_75MB.npy","real_port": "voice_0.05MB.npy", 'proj_port': "proj_6.25MB.npy"},
            },
            "TV": {
                'p2p_pad.json': {"real_port": "voice_0.05MB.npy", 'proj_port': "proj_6.25MB.npy"},
            },
        }

        CONFIG_QOE = {
            "--calc-jitter": list(),
            "--calc-rtt": list(range(6201, 6208)),
        }
        CONTROL = {
            'phone':   {'file_throttle': 0},
            'PC':   {'file_throttle': 0},
            'pad':   {'file_throttle': 0},
            'TV':   {'file_throttle': 0},
        }
    pass




QOS = {
    'phone': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
    'PC': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
    'pad': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
    'TV': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
}

QOE = {
    'jitter': {},
    'rtt': {}
}

data_list = {
    "thu": [],
    "real_len": [],
    "real_len_a": [],
    "real_len_b": [],
    "proj_len": [],
    "qos_list": []
}

init_parameter = {
    'manifest_name': 'manifest.json',
    'stream_idx': 0,
    'port': 0,
    'type': '',
    'tos': 100,
    'cal_rtt': False,
    'no_logging': False
}

FIG = None
LINES = None
AXS = None


def init_figure():
    lines = []
    axs = []
    fig = plt.figure()
    plt.ion()
    for _idx in range(4):
        _ax = fig.add_subplot(221 + _idx)
        _line, = _ax.plot(range(19), 'b-')
        lines.append(_line)
        axs.append(_ax)
    return fig, lines, axs


def update_fig(fig, lines, axs, data_list):
    _lists = [data_list["qos_list"], data_list["thu"],
              data_list["real_len"], data_list["proj_len"]]

    for _idx in range(_lists.__len__()):
        _list = _lists[_idx]
        lines[_idx].set_xdata(range(len(_list)))
        lines[_idx].set_ydata(_list)
        axs[_idx].set_ylim(min(min(_list) * 1.1, 0), max(_list) * 1.1)
        axs[_idx].set_xlim(0, len(_list))
        plt.show()

    fig.canvas.draw()
    fig.canvas.flush_events()

# In this function, the global variable CONFIG will be update with new target_ip


def _ip_extract(keyword):
    names = Connector().list_all()
    clients = [Connector(n) for n in names]

    # extract p2p ip
    conn = Connector()
    ip_table = {}
    for c in clients:
        ip_table.update({c.client: {}})
        if keyword is not None:
            conn.batch(c.client, 'read_ip_addr', {"keyword": keyword})
        else:
            # default extraction
            conn.batch(c.client, 'read_ip_addr', {"keyword": "p2p\|wlan"})

    outputs = conn.executor.wait(1).fetch().apply()
    results = [o['ip_addr'] for o in outputs]

    for r, c in zip(results, clients):
        try:
            ipv4_addrs = eval(r)
        except:
            print("Error: client %s do not exist valid ipv4 addr" % c.client)
        else:
            # print(ipv4_addrs)
            # Generally multiple ipv4 with same name might be detect, but we only need the first one
            for ipv4_addr in ipv4_addrs[::-1]:
                ip_table[c.client].update({ipv4_addr[0]: ipv4_addr[1]})

    # print(ip_table)
    # save the dict into json file
    with open('./temp/ip_table.json', 'w') as f:
        json.dump(ip_table, f)


def setup_ip():
    # read ip from json
    with open("./temp/ip_table.json", "r") as ip_file:
        ip_table = json.load(ip_file)
    for client in CONFIG:
        for _manifest in CONFIG[client]:
            target_device_name = CONFIG[client][_manifest]["receiver"]
            # if target device is server, mianly for test
            if target_device_name == "":
                CONFIG[client][_manifest]["target_addr"] = MONITOR_IP
                continue
            # Deal with p2p and wlan case
            if _manifest.startswith("p2p"):
                CONFIG[client][_manifest]["target_addr"] = ip_table[target_device_name]["p2p"]
            if _manifest.startswith("wlan"):
                # print(ip_table[target_device_name])
                CONFIG[client][_manifest]["target_addr"] = ip_table[target_device_name]["wlan"]
            # Test lo
            if _manifest.startswith("wlp"):
                CONFIG[client][_manifest]["target_addr"] = ip_table[target_device_name]["wlp"]

    # Check the CONFIG ip non empty
    for client in CONFIG:
        for _manifest in CONFIG[client]:
            if CONFIG[client][_manifest]["target_addr"] == "":
                print("Please check the ip_table.json or device connection")
                exit(1)

    print(CONFIG)


def _batch_repara(all_params):
    server_params = []
    client_params = []
    _elementwise_param = {"duration": DURATION}
    for _param in all_params:
        thru_idx = []
        queue_idx = []
        # print(type(_param),_param.copy().update({'port':_param.pop("file_port")}))
        for _manifest in _param:
            if not _manifest.endswith("json"):
                continue
            _elementwise_param.update(
                {"executer": _param["executer"], "target_addr": _param[_manifest]["target_addr"], "receiver": _param[_manifest]["receiver"]})
            for _key in _param[_manifest]:
                if _key.endswith('port'):
                    for _port in _param[_manifest][_key]:
                        _temp_param = _elementwise_param.copy()
                        _temp_param.update(
                            {'port': _port, 'manifest_name': _manifest})
                        server_params.append(_temp_param)
                        if _key == 'file_port':
                            thru_idx.append('file_thru')
                        elif _key == 'real_port':
                            thru_idx.append('real_thru')
                            queue_idx.append('real_len')
                            _temp_param.update({'tos': 128})
                            client_params.append(_temp_param)
                        elif _key == 'proj_port':
                            thru_idx.append('proj_thru')
                            queue_idx.append('proj_len')
                            _temp_param.update({'tos': 128})
                            client_params.append(_temp_param)
        # an indicator maps low level output (only file_thru) to QOS
        _param.update({"thru_idx": thru_idx})
        _param.update({"queue_idx": queue_idx})
    return server_params, client_params


def _transmission_block(conns, params):
    # params: dict[str,list]
    conn = Connector()
    _params = [_param for _param in params]
    # print(_params)
    server_params, client_params = _batch_repara(_params)  # update QOS value
    # print(server_params)
    # print(client_params)
    for _param in server_params:
        if _param['port'] in CONFIG_QOE['--calc-rtt']:
            conn.batch(_param["receiver"], 'outputs_throughput_jitter', {
                       **_param, **{'calc_rtt': '--calc-rtt'}})
            # Since all in AC1, consider 150
        elif _param['port'] in CONFIG_QOE['--calc-jitter']:
            conn.batch(_param["receiver"], 'outputs_throughput_jitter', _param)
        else:
            conn.batch(_param["receiver"], 'outputs_throughput', _param)
    conn.executor.wait(2)

    for _param in _params:
        for _manifest in _param:
            if _manifest.endswith('json'):
                _replay_client = {"manifest_name": _manifest,
                                  "target_addr": _param[_manifest]["target_addr"], "duration": DURATION}
                conn.batch(_param['executer'],
                           'run-replay-client', _replay_client)

    conn.executor.wait(DURATION + 2)
    for _param in client_params:
        conn.batch(_param['executer'], 'compute_queue_length', _param)
    conn.executor.wait(2)
    for _param in client_params:
        if _param['port'] in CONFIG_QOE['--calc-rtt']:
            conn.batch(_param['executer'], 'read_rtt', _param)
    outputs = conn.executor.wait(2).fetch().apply()
    outputs = [n for n in outputs if n]
    # print(outputs)
    # exit()
    _update_QOS_QOE(params, _params, outputs, server_params, client_params)
    return outputs


def _update_QOS_QOE(params, _params, outputs, server_params, client_params):
    for _i in QOS:
        for _j in QOS[_i]:
            QOS[_i][_j] = 0

    # Update QOS, QOE
    _counter = 0
    for _i in range(len(params)):
        _thru_idx = _params[_i]["thru_idx"]
        _key = params[_i]['executer']
        for _j in _thru_idx:
            try:
                _output = ast.literal_eval(outputs[_counter]['file_thru'])
            except:
                print('file_thru', outputs[_counter]['file_thru'])
                raise Exception("Error: {}".format(outputs[_counter]))
            if type(_output) == tuple:
                _port = int(server_params[_counter]['port'])
                QOS[_key][_j] = float(_output[0])
                QOE['jitter'].update({_port: float(_output[1])})
            else:
                QOS[_key][_j] += float(_output)
            _counter += 1

    for _i in range(len(params)):
        queue_idx = _params[_i]["queue_idx"]
        _key = params[_i]['executer']
        for _j in queue_idx:
            QOS[_key][_j] = float(outputs[_counter]["length"])
            _counter += 1

    for _i in range(len(client_params)):
        if client_params[_i]['port'] in CONFIG_QOE['--calc-rtt']:
            QOE['rtt'].update({int(client_params[_i]['port']): float(outputs[_counter]['rtt'])})
            _counter += 1
    return outputs


def _coll_data(tmp_QOS):
    data_list["thu"].append(sum(CALC_THROUGHPUT()))
    data_list["qos_list"].append(float(tmp_QOS["qos"]))
    data_list["real_len"].append((float(
        tmp_QOS["real+file_A"]["real_len"]))/2 + (float(tmp_QOS["real+file_B"]["real_len"]))/2)
    data_list["real_len_a"].append((float(tmp_QOS["real+file_A"]["real_len"])))
    data_list["real_len_b"].append((float(tmp_QOS["real+file_B"]["real_len"])))
    data_list["proj_len"].append(float(tmp_QOS["proj+file"]["proj_len"]))
    pass


def individual_throttle(conns):
    outputs = []
    update_control(conns, [0, 0, 0, 0])
    for _conns in conns:
        output = run_with_throttle([_conns], [0, 0, 0, 0], False)
        outputs.append(float(output[0]['file_thru']))
    return outputs


def compete_throttle(conns):
    outputs = []
    run_with_throttle(conns, [0, 0, 0, 0], False)
    for k in QOS:
        outputs.append(float(QOS[k]['file_thru']))
    return outputs


def measure_mcs(conns, duration):
    print("invalid measure_mcs")
    exit()
    outputs = []
    conn = Connector()
    _param = {'target_addr': PARAMS['target_addr'], 'duration': duration}
    for _conns in conns:
        conn.batch(_conns.client, 'record_mcs', _param)
    outputs = conn.executor.wait(duration + 2).fetch().apply()
    outputs = [float(n['mcs_value']) for n in outputs]
    return outputs


TEST_TOS = 32


def get_conns() -> list:
    conn = Connector()
    clients = conn.list_all()
    conns = [Connector(c) for c in clients]
    for _client in conns:
        # init streams number
        _init_parameters = []
        for _manifest in CONFIG[_client.client]:
            _stream_idx = 0
            if _manifest == "executer":
                continue
            for key in CONFIG[_client.client][_manifest]:
                if not key.endswith("port"):
                    continue
                for _ in CONFIG[_client.client][_manifest][key]:
                    init_parameter['manifest_name'] = _manifest
                    init_parameter['stream_idx'] = _stream_idx
                    init_parameter['port'] = _
                    if _ in CONFIG_QOE['--calc-rtt']:
                        init_parameter['calc_rtt'] = True
                    else:
                        init_parameter['calc_rtt'] = False
                    init_parameter['type'] = CONFIG_FILE[_client.client][_manifest][key]
                    # NOTE:modify the above tos also
                    init_parameter['tos'] = 128 if key == 'proj_port' else TEST_TOS
                    if key == 'real_port':
                        init_parameter['tos'] = 128
                    # if _manifest in ["to_phone_vo.json","speaker.json","to_phone_kb.json"]:
                    # init_parameter['tos'] = 96 if key == 'real_port' else None
                    init_parameter['no_logging'] = False if key == 'proj_port' or key == 'real_port' else True
                    _init_parameters.append(init_parameter.copy())
                    _stream_idx += 1
            conn.batch(_client.client, 'init_stream', {
                       'stream_num': _stream_idx, 'manifest_name': _manifest})
            conn.executor.wait(0.1).apply()
        # init stream parameters
        for _init_parameter in _init_parameters:
            conn.batch(_client.client, 'init_stream_para', _init_parameter)
            conn.executor.wait(0.1)  # give 0.1s for io
        conn.executor.wait(0.1).apply()
    return conns


def _set_priority(conn, name, value):
    for idx in range(PORTS_NUM + 1):
        conn.batch(name, 'set_priority', {'idx': 0, 'priority': value})
    return conn


def update_control(conns, throttle):
    for name, val in zip(CONTROL.keys(), throttle):
        CONTROL[name]['file_throttle'] = val
    # apply throttle control
    conn = Connector()
    for _client in conns:
        if _client.client in CONTROL.keys():
            for _manifest in CONFIG[_client.client]:
                # FIXME: need to change throttle for individual manifest
                if "file_port" in CONFIG[_client.client][_manifest]:
                    # print({**CONTROL[_client.client], **{"manifest_name":_manifest}})
                    conn.batch(_client.client, 'throttle', {
                               **CONTROL[_client.client], **{"manifest_name": _manifest}})
    conn.executor.wait(0.01).apply()
    pass


def run_with_throttle(conns, throttle, is_save=True):
    # print('throttle\t', sum(throttle))
    update_control(conns, throttle)
    all_params = [{**CONFIG[n.client], "executer": n.client} for n in conns]
    while True:
        try:
            outputs = _transmission_block(conns, all_params)
            break
        except Exception as e:
            traceback.print_exc()
            # raise(e)
            # print(e)
            continue

    # calculate QoS
    qos_proj = PROJ_LEN_WEIGHT * \
        sum([float(x['proj_len']) for x in QOS.values()])
    qos_real = REAL_LEN_WEIGHT * \
        sum([float(x['real_len']) for x in QOS.values()])
    qos_file = -THRU_WEIGHT * sum([float(x['file_thru'])
                                  for x in QOS.values()])
    qos = qos_proj + qos_real + qos_file
    # save data
    tmp: dict[str, dict[str, int]] = {**QOS.copy(), **QOE}
    tmp['throttle'] = throttle
    tmp['qos'] = qos

    if FIG is not None:
        update_fig(FIG, LINES, AXS, data_list)
    if is_save:
        f.write(json.dumps(tmp)+',')
        f.flush()
    return qos


def one_dimentional_search(initial_point, ref_queue_length, step_size):
    # find device and manifest with proj_port
    obj_clients = []
    obj_ports = []
    for client in CONTROL.keys():
        for manifest in CONFIG[client]:
            if "proj_port" in CONFIG[client][manifest] and len(CONFIG[client][manifest]["proj_port"]) != 0:
                obj_clients.append(client)
                # extract the projection element, the only element in the list
                obj_ports.append(CONFIG[client][manifest]["proj_port"][0])
    # print(obj_ports)
    if not obj_clients:
        print("no valid target stream")
        exit()
    throttle = np.asarray(initial_point)
    conns = ORDERED_CONNECTOR()

    # Simulation parameter

    # initial point is a vector length match the number of controllable clients
    trial = 3
    history_rtt = 0
    while True:
        # get the current throttle
        # run with throttle
        temp_QOE = None
        temp_QOS = None
        temp_qos = 0
        print(throttle)
        # update CONFIG stream num
        # file_num = throttle/file_throttle_each

        # qos = run_with_throttle(conns, [0,0], is_save=False)
        running_idx = 0
        # while True:
        for _ in range(trial):
            running_idx += 1
            # disable save for multiple running
            qos = run_with_throttle(conns, throttle.tolist(), is_save=False)
            thrus = CALC_THROUGHPUT()
            if temp_QOE is not None:
                rtt_value = 0
                # Pick maximum rtt value
                for obj_port in obj_ports:
                    _rtt_value = temp_QOE["rtt"][obj_port] * 1000
                    rtt_value = _rtt_value if rtt_value < _rtt_value else rtt_value
                if rtt_value < history_rtt:
                    history_rtt = rtt_value
                    temp_QOE = QOE.copy()
                    temp_QOS = QOS.copy()
                    temp_qos = qos
            else:
                temp_QOE = QOE.copy()
                temp_QOS = QOS.copy()
                temp_qos = qos
                rtt_value = 0
                for obj_port in obj_ports:
                    _rtt_value = temp_QOE["rtt"][obj_port] * 1000
                    rtt_value = _rtt_value if rtt_value < _rtt_value else rtt_value
                history_rtt = rtt_value
            # print(rtt_value)

            # for obj_port in obj_ports:
            # print("running time:\t", running_idx)
            # print(temp_QOS)
            # print(temp_QOE)
            rtt_value = 0
            for obj_port in obj_ports:
                _rtt_value = temp_QOE["rtt"][obj_port] * 1000
                rtt_value = _rtt_value if rtt_value < _rtt_value else rtt_value
            if rtt_value <= ref_queue_length:
                break

        # save the qos value for multiple running
        tmp: dict[str, dict[str, int]] = {**temp_QOS.copy(), **temp_QOE}
        tmp['throttle'] = throttle.tolist()
        tmp["qos"] = temp_qos
        f.write(json.dumps(tmp)+',')
        f.flush()


        # displace saved QOS ans QOE



        # extract queue information from QOS
        # queue_length = temp_QOS[obj_client]["proj_len"]
        rtt_value = 0
        for obj_port in obj_ports:
            _rtt_value = temp_QOE["rtt"][obj_port] * 1000
            rtt_value = _rtt_value if rtt_value < _rtt_value else rtt_value
        # print(rtt_value)
        direction = int(rtt_value <= ref_queue_length)
        thrus = CALC_THROUGHPUT()

        print( json.dumps(temp_QOS, indent=2) )
        print( json.dumps(temp_QOE, indent=2) )

        print("Throughput\t", sum(thrus))
        print("="*50)
        if rtt_value <= ref_queue_length:
            break
        if direction == 1:
            break

        if 'history_direction' in locals() and history_direction != direction:
            step_size = step_size * 0.5
            print("step_size", step_size)

        # get the new throttle
        throttle = throttle + step_size * 2 * (direction - 0.5)
        # stop history direction
        history_direction = direction


def individual_task():
    global CONFIG, QOS, QOE
    # take a deep copy of CONFIG
    import copy
    orignal_CONFIG = copy.deepcopy(CONFIG)
    # take a deep copy of QOE, QOE
    saved_QOS = copy.deepcopy(QOS)
    saved_QOE = copy.deepcopy(QOE)
    
    #
    conns = ORDERED_CONNECTOR()

    # Initial empty CONFIG File
    for client in orignal_CONFIG.keys():
        for manifest in orignal_CONFIG[client]:
            for _port in orignal_CONFIG[client][manifest]:
                if _port.endswith("port"):
                    CONFIG[client][manifest][_port] = []
    
    # Individual Test
    for client in orignal_CONFIG.keys():
        for manifest in orignal_CONFIG[client]:
            for _port in orignal_CONFIG[client][manifest]:
                temp_QOS = None
                temp_QOE = None
                QOE = {'jitter': {},'rtt': {}}
                QOS = {
                    'phone': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
                    'PC': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
                    'pad': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
                    'TV': {'file_thru': 0, 'real_thru': 0, 'proj_thru': 0, 'real_len': 0, 'proj_len': 0},
                }
                if _port.endswith("port"):
                    CONFIG[client][manifest][_port] = orignal_CONFIG[client][manifest][_port]
                    get_conns()
                    # Select the best results
                    print(CONFIG)
                    for _trial in range(3):
                        run_with_throttle(conns, [0, 0, 0], False)
                        if temp_QOS is None:
                            temp_QOS = copy.deepcopy(QOS)
                            temp_QOE = copy.deepcopy(QOE)
                        else:
                            if _port == "file_port":
                                if temp_QOS[client]["file_thru"] < QOS[client]["file_thru"]:
                                    temp_QOS = copy.deepcopy(QOS)
                                    temp_QOE = copy.deepcopy(QOE)
                            else:
                                for _key in temp_QOE["rtt"].keys():
                                    if temp_QOE["rtt"][_key] > QOE["rtt"][_key]:
                                        temp_QOS = copy.deepcopy(QOS)
                                        temp_QOE = copy.deepcopy(QOE)
                        # print("=======================")
                        # print(QOS)
                        # print(QOE)
                    # update QOS and QOE to saved one
                    saved_QOS = {key1: {
                        key2: temp_QOS[key1][key2] + saved_QOS[key1][key2] for key2 in temp_QOS[key1]} for key1 in temp_QOS}

                    [saved_QOE[key].update(temp_QOE[key]) for key in temp_QOE]
                    CONFIG[client][manifest][_port] = []
                    print("=======================")
                    print(temp_QOS)
                    print(temp_QOE)
    # Reset CONFIG
    CONFIG = orignal_CONFIG.copy()
    print("=======================")
    print( json.dumps(saved_QOS, indent=2))
    print( json.dumps(saved_QOE, indent=2))
    with open(f'./temp/individual_task_{time.ctime()}.json', 'a+') as fw:
        fw.write(json.dumps(saved_QOS)+',')
        fw.write(json.dumps(saved_QOE))
        fw.flush()


def system_test(conns):

    step_size = 10
    throttle = np.asarray([0, 0, 0])

    while True:
        _trial = 0
        try:
            print("-------------START------------")
            while _trial < 3:
                qos = run_with_throttle(conns, throttle.tolist())
                thrus = CALC_THROUGHPUT()
                print( json.dumps(QOS, indent=2) )
                print( json.dumps(QOE, indent=2) )
                print("Control\t", sum(throttle))
                print("Throughput\t", sum(thrus))
                print("QOS Value\t", qos)
                _trial += 1
            break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            continue


epsilon_M = 0.5
e_I = 1
h_K = 1
alpha = 5
lower_bound = np.array([0, 0, 0, 0])
upper_bound = np.array([250, 250, 250, 250])
is_plot = True
starve_value = 10


def gradient_search(conns):
    global FIG, LINES, AXS
    if is_plot:
        FIG, LINES, AXS = init_figure()
    counter = 0

    x_his = np.array([60, 60, 60, 60]) - 10
    # x = np.array([10, 10, 10, 10])
    x = np.array(compete_throttle(conns))
    x = x - 20
    x = np.asarray([_ if _ > starve_value else starve_value for _ in x])
    # x = np.asarray( measure_mcs(conns, 5) )
    # x = x / len(x)
    x_his = x + 5
    print(x)
    ##
    while True:
        counter += 1
        direction = counter % len(x)
        h_fd = max(1, abs(x[direction]))*epsilon_M
        ##
        temp_x = x + h_fd * e_I

        print(x, temp_x)
        qos = run_with_throttle(conns, x.tolist())
        print('QOE', QOE)
        qos_h = run_with_throttle(conns, temp_x.tolist(), False)
        print(qos, qos_h)
        ##
        delta_x = alpha * h_K * (qos_h - qos) / h_fd
        x_ = x - delta_x

        if True in (x_ <= lower_bound) or True in (x_ >= upper_bound):
            h_delta_x = (x - x_his)*0.5
            h_direction = np.sign(-delta_x)
            x_ = x + h_direction*abs(h_delta_x)
            x_his = x
        x = x_
        pass
    pass


def _test(_conns, throttle):
    while True:
        try:
            output = run_with_throttle([_conns], throttle, True)
            thrus = CALC_THROUGHPUT()
            print(QOE)
            print("Throughput\t", sum(thrus))
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            # print(e)
            break
            # continue


def main(args):
    optimize_scenario(args.scenario)
    threshold = [18,20,18]
    start_point = [750, 350 ,550]
    step_size = [50,25,50]
    # Initialize manifest config
    conns = get_conns()
    conns = ORDERED_CONNECTOR()
    # Set ip for CONFIG
    _ip_extract("wlan\|p2p\|wlp")
    setup_ip()
    if args.test == 1:
        system_test(conns)
    elif args.test == 2:
        individual_task()
    else:
        control_throttle = start_point[args.scenario - 1]
        step_size = step_size[args.scenario - 1]
        if args.scenario == 3:
            control_throttles = [control_throttle ,0, (control_throttle - 89) / 1.39,0]
            step_sizes = [step_size, 0 ,   step_size / 1.39,0 ]
        else:
            control_throttles = [control_throttle, control_throttle]
            step_sizes = [step_size,step_size]
        one_dimentional_search(np.array(control_throttles), threshold[args.scenario - 1], step_size = np.array(step_sizes))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenario', type=int,
                        default=1, help='scenario in 1,2,3')
    parser.add_argument('-t', '--test', type=int, default=0,
                        help='start test mode, test = 0 -> one dimentional search, test = 1 -> system test, test = 2 -> individual test')

    args = parser.parse_args()
    main(args)
