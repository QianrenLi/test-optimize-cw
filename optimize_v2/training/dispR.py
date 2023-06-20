import matplotlib.pyplot as plt
import numpy as np

def dispLoss(path, agg_num, is_save = False):
    losses = []
    loss = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        # iterative over lines with index
        for i, line in enumerate(lines):
            loss += float(line)
            if i % agg_num == 0:
                losses.append(loss / agg_num)
                loss = 0

    indices = list(range(0, len(losses) * agg_num, agg_num))
    plt.plot(indices,losses)
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.show()
    if is_save:
        plt.savefig("%s.png" % path[:-4])

def dispValue(path, agg_num, gamma , is_save = False):
    values = []
    legends = []

    value = 0
    j = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        i = 0
        average_counter = 0
        # iterative over lines with index
        for line in lines:
            if line != "\n":
                cost = float(line)
                i += 1
                average_counter += 1
            else:
                value = 0
                average_counter = 0
                values = np.array(values)
                indices = list(range(0, len(values) * agg_num, agg_num))
                legends.append("Scenario %d" % j)
                j += 1
                plt.plot(indices,values)
                values = []
                continue
            value += cost
            if i % agg_num == 0:
                values.append(value / average_counter)
                

    
    plt.xlabel("Training Iteration")
    plt.ylabel("Value")
    plt.legend(legends)
    plt.title("Average Cost per Scenario")
    plt.show()
    if is_save:
        plt.savefig("%s.png" % path[:-4])


if __name__ == '__main__':
    # loss_path = f"logs/log-loss-2023-06-18-16:08:06.txt"
    # cost_path = f"logs/log-cost-2023-06-18-16:08:05.txt"
    # loss_path = f"logs/log-loss-2023-06-18-15:39:56.txt"
    # cost_path = f"logs/log-cost-2023-06-18-15:39:55.txt"
    # loss_path = f"logs/log-loss-2023-06-20-17:47:01.txt"
    # cost_path = f"logs/log-cost-2023-06-20-17:47:00.txt"
    # loss_path = f"logs/log-loss-2023-06-20-20:27:05.txt"
    # cost_path = f"logs/log-cost-2023-06-20-20:27:04.txt"
    loss_path = f"logs/log-loss-2023-06-20-21:05:08.txt"
    cost_path = f"logs/log-cost-2023-06-20-21:05:07.txt"
    dispLoss(loss_path,1)
    dispValue(cost_path, 1, 0.9)

