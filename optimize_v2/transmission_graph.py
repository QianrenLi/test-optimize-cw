#!/usr/bin/env python3
import json

##=======================================================================##
import ctypes
import os
os.system('make')
NATIVE_MOD = ctypes.CDLL('./liboptimize.so')
NATIVE_MOD.fraction_to_throttle.restype = ctypes.c_void_p
NATIVE_MOD.fraction_to_throttle.argtypes = [
    ctypes.c_float, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float) ]
def _list_to_c_array(arr: list, arr_type=ctypes.c_float):
    return (arr_type * len(arr))(*arr)
##=======================================================================##

# define the graph class
class Graph:
    graph = {}
    info_graph = {}

    def __init__(self):
        self.trans_graph = dict()
        pass

    def ADD_DEVICE(self, device_name):
        self.graph.update({device_name: {}})
        self.info_graph.update({device_name: {}})
        pass

    def REMOVE_DEVICE(self, device_name):
        del self.trans_graph[device_name]

    def ADD_LINK(self, device_name, target_name, protocol, MCS):  # -> link name
        link_name = protocol+'_'+device_name+'_'+target_name
        self.graph[device_name].update({link_name: {}})
        self.info_graph[device_name].update({link_name: {'MCS': MCS}})
        return link_name

    def REMOVE_LINK(self, link_name):
        device_name = link_name.split('_')[1]
        del self.trans_graph[device_name][link_name]

    def ADD_STREAM(self, link_name, port_number, file_name, thru, duration, tos=32, target_rtt=18, name = ''):
        # from link name to device name
        device_name = link_name.split('_')[1]
        if type(port_number) == list:
            for _port_number in port_number:
                _name = name if name != '' else str(_port_number)+'@'+str(tos)
                self.graph[device_name][link_name].update({str(_port_number)+'@'+str(tos): {
                                                          'file_name': file_name, 'thru': thru, 'throughput': '', "throttle": 0, 'duration': duration}})
                self.info_graph[device_name][link_name].update(
                    {str(_port_number)+'@'+str(tos): {"target_rtt": target_rtt, 'name': _name}})
        else:
            _name = name if name != '' else str(port_number)+'@'+str(tos)
            self.graph[device_name][link_name].update({str(port_number)+'@'+str(tos): {
                                                      'file_name': file_name, 'thru': thru, 'throughput': '', "throttle": 0, 'duration': duration}})
            self.info_graph[device_name][link_name].update(
                {str(port_number)+'@'+str(tos): {"target_rtt": target_rtt, 'name': _name}})
        pass

    def REMOVE_STREAM(self, link_name, port_number, tos=132):
        device_name = link_name.split('_')[1]
        del self.graph[device_name][link_name][str(port_number)+'@'+str(tos)]
        del self.info_graph[device_name][link_name][str(
            port_number)+'@'+str(tos)]

    def UPDATE_DURATION(self, link_name, port_number, duration):
        device_name = link_name.split('_')[1]
        self.graph[device_name][link_name][port_number]['duration'] = duration

    def associate_ip(self, device_name, protocol, ip_addr):
        self.info_graph[device_name].update({protocol+'_ip_addr': ip_addr})
        pass

    def show(self):
        print(json.dumps(self.info_graph, indent=2))
        print("="*50)
        print(json.dumps(self.graph, indent=2))
        pass

    # After getting reply, the stream might be deleted
    def update_graph(self, reply):
        for device_name, links in self.graph.items():
            for link_name, streams in list(links.items()):
                for port_name in list(streams.keys()):
                    try:
                        streams[port_name].update(reply[link_name][port_name])
                        self.info_graph[device_name][link_name][port_name].update({
                                                                                  'active': True})
                    except:
                        # if port_name not in reply[link_name].keys():
                        self.info_graph[device_name][link_name][port_name].update(
                            {'active': False})
                    # else:
        pass

    def graph_to_control_coefficient(self):
        thru = {}
        throttle = {}
        mcs = {}
        # link throughput calculation and throttle detection
        for device_name, links in self.graph.items():
            for link_name, streams in links.items():
                link_thru = 0
                for port_number, stream in streams.items():
                    link_thru += stream['thru']
                    # detect file by keyword "file" in name or can be detected by thru value
                    if stream['thru'] == 0:
                        throttle.update({link_name: stream['thru']})
                thru.update({link_name: link_thru})
                mcs.update(
                    {link_name: self.info_graph[device_name][link_name]['MCS']})
        return thru, throttle, mcs

    # core function
    @staticmethod
    def _update_throttle(sorted_mcs, sorted_thru, allocated_times):
        # calculate the throughput/MCS (without file) of each link
        link_fraction = sum( [sorted_thru[i]/sorted_mcs[i] for i in range(len(sorted_mcs))] )
        # calculate the normalized throughput
        normalized_thru = ( link_fraction + allocated_times ) / len(sorted_mcs);
        return [normalized_thru * sorted_mcs[i] - sorted_thru[i] for i in range(len(sorted_mcs))]

    @staticmethod
    def _init_allocated_times(sorted_mcs, sorted_thru, init_factor):
        allocated_times = 1 - sum( [sorted_thru[i]/sorted_mcs[i] for i in range(len(sorted_mcs))] )
        return allocated_times*init_factor

    def _link_to_port_throttle(self, link_throttle):
        port_throttle = {}
        for device_name, links in self.graph.items():
            for link_name, streams in links.items():
                if link_name in link_throttle.keys():
                    port_throttle.update({link_name: {}})
                    # calculate file number
                    file_number = 0
                    for port_name, stream in streams.items():
                        if stream["thru"] == 0:
                            file_number += 1
                    # calculate port throttle
                    _port_throttle = link_throttle[link_name] / (file_number)
                    for port_name, stream in streams.items():
                        if stream["thru"] == 0:
                            stream.update({"throttle": _port_throttle})
                            port_throttle[link_name].update(
                                {port_name: _port_throttle})
        return port_throttle

    def update_throttle(self, fraction, reset_flag:bool):
        # add last throttle value together
        thru, throttle, mcs = self.graph_to_control_coefficient()
        sorted_mcs = [mcs[key] for key in throttle]
        sorted_thru = [thru[key] for key in throttle]
        ##
        length = len(throttle)
        sorted_mcs = _list_to_c_array( sorted_mcs)
        sorted_thru = _list_to_c_array( sorted_thru )
        out_sorted_throttle = _list_to_c_array( [0.0]*length )
        ##
        if reset_flag:
            fraction = NATIVE_MOD.init_throttle_fraction(length, sorted_mcs, sorted_thru)
        ##
        NATIVE_MOD.fraction_to_throttle(fraction, length, sorted_mcs, sorted_thru,
                                        out_sorted_throttle)
        out_sorted_throttle = [float(x) for x in out_sorted_throttle]

        # out_sorted_throttle = self._update_throttle([mcs[key] for key in throttle], [thru[key] for key in throttle], fraction)
        print("reset_flag",reset_flag)
        print("out_sorted_throttle",out_sorted_throttle)        
        for i, link_name in enumerate(throttle.keys()):
            throttle.update({link_name: out_sorted_throttle[i]})
        ##
        port_throttle = self._link_to_port_throttle(throttle)
        return port_throttle
