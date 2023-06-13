import matplotlib.pyplot as plt

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
    value = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        # iterative over lines with index
        for i, line in enumerate(lines):
            cost =  float(line)
            value = gamma * value + cost
            if i % agg_num == 0:
                values.append(value)

    indices = list(range(0, len(values) * agg_num, agg_num))
    plt.plot(indices,values)
    plt.xlabel("Training Iteration")
    plt.ylabel("Value")
    plt.title("Value function starts at ${S_0}$")
    plt.show()
    if is_save:
        plt.savefig("%s.png" % path[:-4])


if __name__ == '__main__':
    loss_path = f"logs/log-loss-2023-06-08-21:20:32.txt"
    cost_path = f"logs/log-cost-2023-06-08-21:20:30.txt"
    # dispLoss(loss_path,1)
    dispValue(cost_path, 1, 0.9)

