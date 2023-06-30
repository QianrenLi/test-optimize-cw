
from training.txAgent import tx
from training.txController import txController
from training.environment import envCap
from tap_pipe import ipc_socket
from transmission_graph import Graph

import test_case as tc
import threading


is_stop = threading.Event()


def add_file(wireless_tx: tx, link_name:str, port:int):
    wireless_tx.graph.ADD_STREAM(
        link_name,
        port_number=port,
        file_name="file_75MB.npy",
        duration=[0, wireless_tx.DURATION],
        thru=0,
        tos=96,
        name="File",
    )    


def main():
    ## Setup graph
    graph,lists = tc.cw_training_case()


    ## Setup environment
    env = envCap(graph)

    ## Setup tx object
    wireless_tx = tx(graph)
    add_file(wireless_tx, lists[0], 6200)
    ## Set

    ## Setup sockets
    ctl_prot = "lo"
    wireless_tx.prepare_transmission(ctl_prot)

    ## Setup action space
    action_space = [
        [i / 20 for i in range(1, 20, 1)],
        [1, 3 , 7 , 15, 31, 63],  # CW value
        [1, 3, 5, 7, 9, 11]
    ]

    ## Setup txController object
    wireless_controller = txController(graph, is_stop, wireless_tx.socks, action_space)
    wireless_controller.graph.show()

    wireless_tx.transmission_thread()

if __name__ == "__main__":
    main()