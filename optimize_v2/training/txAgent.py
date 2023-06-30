import os
import json
import sys
import ctypes
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from transmission_graph import Graph
from tap import Connector
from tap_pipe import ipc_socket


class tx:
    def __init__(self, graph: Graph):
        self.init_thottle = 30
        self.DURATION = 10
        self.is_local_test = False
        self.is_line_search_policy = False
        self.heuristic_fraction = 0.1

        self.graph = graph
        self.socks = []

    ## ip setup component
    def _ip_extract(self, keyword: str):
        """
        Extract ip from controlled device, store ip table into temp/ip_table.json
        """
        conn = Connector()
        ip_table = {}
        for device_name, links in self.graph.graph.items():
            if keyword is not None:
                conn.batch(device_name, "read_ip_addr", {"keyword": keyword})
            else:
                conn.batch(device_name, "read_ip_addr", {"keyword": "p2p\\|wlan"})
        outputs = conn.executor.wait(1).fetch().apply()
        results = [o["ip_addr"] for o in outputs]
        for r, c in zip(results, self.graph.graph.keys()):
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

        return self

    def _setup_ip(self):
        """
        Set up ip stored in graph by the ip_table (requirement to setup the transmission)
        """
        with open("./temp/ip_table.json", "r") as ip_file:
            ip_table = json.load(ip_file)

        for device_name in ip_table.keys():
            _depart_name = device_name.split("-")
            if len(_depart_name) > 1:
                _, ind = device_name.split("-")
                for protocol, ip in ip_table[device_name].items():
                    self.graph.info_graph[device_name].update({"ind": int(ind)})
                    self.graph.associate_ip(device_name, protocol[0:3], ip)                  # default take the previous three as indicator to protocol
            else:
                for protocol, ip in ip_table[device_name].items():
                    self.graph.associate_ip(device_name, protocol[0:3], ip)
        return self
        # graph.show()

    ## ipc communication component
    def _add_ipc_port(self):
        """
        Add ipc port (remote and local) to graph
        """
        port = 8888
        for device_name in self.graph.graph.keys():
            for link_name in self.graph.graph[device_name].keys():
                self.graph.info_graph[device_name][link_name].update({"ipc_port": port})
                self.graph.info_graph[device_name][link_name].update(
                    {"local_port": port - 1024}
                )
                port += 1
        return self

    ## transmission component
    def _set_manifest(self):
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
        for device_name, links in self.graph.graph.items():

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
                        parameter.update({"throttle": self.init_thottle})
                    parameter.update({"start": stream["duration"][0]})
                    parameter.update({"stop": stream["duration"][1]})
                    _init_parameters.append(parameter)
                # write detailed to device
                for i, _parameter in enumerate(_init_parameters):
                    conn.batch(
                        device_name,
                        "init_stream_para",
                        {**_parameter, **{"stream_idx": i}},
                    )
                    print({**_parameter, **{"stream_idx": i}})
                    conn.executor.wait(0.5)
                conn.executor.wait(0.5).apply()
        return self

    def _transmission_block(self):
        """
        Construct a transmission ready connector waiting to be applied
        """
        conn = Connector()
        # start reception
        for device_name, links in self.graph.graph.items():
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
                            {"port": port_num, "duration": self.DURATION},
                            timeout=self.DURATION + 5,
                        )
                    else:
                        conn.batch(
                            receiver,
                            "outputs_throughput_jitter",
                            {
                                "port": port_num,
                                "duration": self.DURATION,
                                "calc_rtt": "--calc-rtt",
                                "tos": tos,
                            },
                            timeout=self.DURATION + 5,
                        )

        conn.executor.wait(1)
        # start transmission
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                # split link name to protocol, sender, receiver
                prot, sender, receiver = link_name.split("_")
                print("receiver", receiver)
                if receiver:
                    ip_addr = self.graph.info_graph[receiver][prot + "_ip_addr"]
                else:
                    ip_addr = "192.168.3.37" if not self.is_local_test else "127.0.0.1"
                conn.batch(
                    sender,
                    "run-replay-client",
                    {
                        "target_addr": ip_addr,
                        "duration": self.DURATION,
                        "manifest_name": link_name + ".json",
                        "ipc-port": self.graph.info_graph[sender][link_name][
                            "ipc_port"
                        ],
                    },
                    timeout=self.DURATION + 5,
                )

        return conn.executor.wait(self.DURATION + 5)

    def _loop_apply(self, conn):
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

    def _rtt_port_associate(self, outputs):
        """
        Associate rtt value lists to the corresponding ports id (for better vision)
        """
        rtt_value = {}
        rtt_list = []
        idx = 0
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                prot, sender, receiver = link_name.split("_")
                for stream_name, stream in streams.items():
                    port, tos = stream_name.split("@")
                    if stream["thru"] != 0:
                        rtt_value.update({stream_name: float(outputs[idx]["rtt"])})
                        rtt_list.append(float(outputs[idx]["rtt"]))
                        idx += 1
        print(np.round(np.array(rtt_list) * 1000, 3))
        return rtt_value

    ## Result preprocessing component
    def _sum_file_thru(self, outputs):
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

    def _calc_rtt(self):
        """
        Construct a rrt calculation ready connector waiting to be applied
        """
        conn = Connector()
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                prot, sender, receiver = link_name.split("_")
                for stream_name, stream in streams.items():
                    port, tos = stream_name.split("@")
                    if stream["thru"] != 0:
                        conn.batch(sender, "read_rtt", {"port": port, "tos": tos}).wait(
                            0.1
                        )
        return conn.executor.wait(0.5)

    def setup_sockets(self, ctl_prot):
        if len(self.socks) > 0:
            [sock.close() for sock in self.socks]
        self.socks.clear()
        for device_name, links in self.graph.graph.items():
            for link_name, streams in links.items():
                # start threads to send data
                prot, sender, receiver = link_name.split("_")
                ip_addr = self.graph.info_graph[sender][ctl_prot + "_ip_addr"]               ## since no p2p/lo is considered
                sock = ipc_socket(
                    ip_addr,
                    self.graph.info_graph[device_name][link_name]["ipc_port"],
                    local_port=self.graph.info_graph[device_name][link_name]["local_port"],
                    link_name=link_name,
                )
                self.socks.append(sock)
        return self

    def prepare_transmission(self, ctrl_prot):
        """
        Prepare the transmission by setting up the ip address, ipc port, and manifest
        """
        self._ip_extract(
            "wlan\\|p2p\\|wlx\\|wlp"
        )._setup_ip()._add_ipc_port().setup_sockets(ctrl_prot)._set_manifest()
        return self

    def transmission_thread(self):
        conn = self._transmission_block()
        results = self._sum_file_thru(self._loop_apply(conn))
        conn = self._calc_rtt()
        print(self._rtt_port_associate(self._loop_apply(conn)))
        return results
