from system2trainV2 import wlanDQNController
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_network(controller: wlanDQNController):
    state_num = controller.max_agent_num * controller.agent_states_num

    collected_controls = [[]]

    for cw in [controller.cw_levels[3]]:
        for aifs in[ controller.aifs_levels[3]]:
            for rtt_i in range(0,180):
                rtt = rtt_i / 10
                state = np.zeros(state_num)
                state[0] = rtt
                state[1] = cw
                state[2] = aifs
                # collect_controls["cw_"].append(cw)
                # collect_controls["aifs_"].append(aifs)
                controls, _ = controller.get_action(state)
                for idx, value in enumerate(list(controls[0])): 
                    if len(collected_controls) <= idx + 1:
                        collected_controls.append([])
                    collected_controls[idx].append(value)
                # for key in controls:
                #     if key not in collect_controls:
                #         collect_controls.update({key: []})
                #     collect_controls[key].append(controls[key])
    for values in collected_controls:
        plt.plot(values)
        plt.show()
    # for key in collect_controls:
    #     if key != "rtt":
    #         plt.plot(collect_controls["rtt"], collect_controls[key])
    # plt.legend(["cw", "aifs"])
    # print(collect_controls)

import test_case as tc
graph,lists = tc.cw_training_case()
graph,_ = tc.cw_training_case()
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
for lnk in lists[0:1]: 
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

abs_path = os.path.dirname(os.path.abspath(__file__))
wlanController.load_params(abs_path + "/model/6-26-3.pt")
# print(abs_path)
evaluate_network(wlanController)