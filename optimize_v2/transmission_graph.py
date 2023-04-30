import numpy as np
import json
from tap import Connector

# define the graph class


class Graph:
    graph = {}
    info_graph = {}

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

    def ADD_STREAM(self, link_name, port_number, file_name, thru, duration, tos=32, target_value=18):
        # from link name to device name
        device_name = link_name.split('_')[1]
        if type(port_number) == list:
            for _port_number in port_number:
                self.graph[device_name][link_name].update({str(_port_number)+'@'+str(tos): {
                                                          'file_name': file_name, 'thru': thru, 'throughput': '', "throttle": 0, 'duration': duration}})
                self.info_graph[device_name][link_name].update(
                    {str(_port_number)+'@'+str(tos): {"target_value": target_value}})
        else:
            self.graph[device_name][link_name].update({str(port_number)+'@'+str(tos): {
                                                      'file_name': file_name, 'thru': thru, 'throughput': '', "throttle": 0, 'duration': duration}})
            self.info_graph[device_name][link_name].update(
                {str(port_number)+'@'+str(tos): {}})
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
    def _compute_throttle(self, sorted_mcs, sorted_thru, allocated_times):
        matrix_size = len(sorted_mcs)
        matrix_A = np.zeros((matrix_size, matrix_size))
        vetor_b = np.zeros(matrix_size)
        for i in range(matrix_size):
            if i == 0:
                matrix_A[i, :] = 1 / np.array(sorted_mcs)
                vetor_b[i] = allocated_times
            else:
                matrix_A[i, i-1:i +
                         1] = np.array([sorted_mcs[i-1], -sorted_mcs[i]])
                vetor_b[i] = sorted_thru[i]/sorted_mcs[i] - \
                    sorted_thru[i]/sorted_mcs[i-1]

        # solve linear equations denoted by matrix A and vector b
        throttle = np.linalg.solve(matrix_A, vetor_b)
        return throttle

    def _set_throttle(self, thru, throttle, mcs, start_fraction):
        sorted_mcs = [mcs[key] for key in throttle]
        sorted_thru = [thru[key] for key in throttle]

        sorted_throttle = self._compute_throttle(
            sorted_mcs, sorted_thru, start_fraction)
        for i, link_name in enumerate(throttle.keys()):
            throttle.update({link_name: sorted_throttle[i]})
        return throttle

    def _update_throttle(self, thru, throttle, mcs, fraction):
        throttle = self._set_throttle(thru, throttle, mcs, fraction)
        return throttle

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

    def update_throttle(self, last_fraction, step_size):
        '''
            step_size: file throughput changes in the system
        '''
        # add last throttle value together
        thru, throttle, mcs = self.graph_to_control_coefficient()
        fraction = min(0.99, last_fraction +
                       step_size) if last_fraction + step_size > 0 else 0.0001
        link_throttle = self._update_throttle(thru, throttle, mcs, fraction)
        port_throttle = self._link_to_port_throttle(link_throttle)
        return fraction, port_throttle
