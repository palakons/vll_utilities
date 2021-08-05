import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size


def plot_gpu_utilization(
    time_steps,
    gpu_util,  # shape (len(node_names) * n_gpu_per_node, len(time_steps))
    n_gpu_per_node=4,
    node_names=[f"v{i+1:02d}" for i in range(17)],
    target_path=None,
):

    point_step = (max(time_steps) - min(time_steps)) / (len(time_steps) - 1)
    n_node = len(node_names)

    title = f"VLL Cluster GPU Utilization History: {min(time_steps)} to {max(time_steps)} UTC+0 time"

    fig, axs = plt.subplots(n_node * n_gpu_per_node, sharex=True, figsize=(10, 10))
    plt.xticks(rotation=15)
    axs[0].set_title(title)
    my_cmap = plt.get_cmap("RdYlGn_r")
    rescale = lambda y: (y - 0) / (100 - 0)

    for i, ax in enumerate(axs):
        ax.grid(which="major", axis="x")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        if i < len(axs) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        if (i + 1) % n_gpu_per_node != 0:
            ax.spines["bottom"].set_visible(False)
        else:
            d = 0.88 - 0.107 - 0.0055
            fig.text(
                0.095,
                1
                - (i // n_gpu_per_node * d / n_node + 0.107 + 0.015 + d / (2 * n_node)),
                node_names[i // n_gpu_per_node],
                ha="center",
                va="center",
            )
        # print(len(time_steps),len(gpu_util[i]))
        ax.bar(
            time_steps,
            gpu_util[i],
            width=point_step,
            color=my_cmap(rescale(gpu_util[i])),
        )

    plt.subplots_adjust(hspace=0)
    # plt.axis('tight')
    plt.xlabel("time")
    fig.text(0.05, 0.5, "GPU utilization", ha="center", va="center", rotation=90)
    if target_path is not None:
        fig.savefig(target_path)
    return fig


def process_util_string(line):
    data = {}
    pattern = r" \| "
    mod_string = re.sub(pattern, "", line)
    pattern = r","
    mod_string = re.sub(pattern, "", mod_string)
    pattern = r"[a-z_]+ {2}"
    mod_string = re.sub(pattern, "", mod_string)
    token = mod_string.split()
    # print(token)
    current_token = ""
    for tag in token:
        if tag[0] == "v":
            current_token = tag
        else:
            if current_token not in data:
                data[current_token] = [int(tag)]
            else:
                data[current_token].append(int(tag))
    # print(data)
    return data


def pad_zero_util(gpu_util_list, n_gpu_per_node=4):
    if len(gpu_util_list) < n_gpu_per_node:
        return gpu_util_list + [0] * (n_gpu_per_node - len(gpu_util_list))
    return gpu_util_list[:n_gpu_per_node]

def combine_gpu_util(node_names, data, n_gpu_per_node):
    result = []
    for name in node_names:
        result += pad_zero_util(data[name] if name in data else [], n_gpu_per_node)
    return result


def read_gpu_log(
    input_file="/home2/palakons/track_cluster_usage.log",
    n_gpu_per_node=4,
):

    t = []

    data_nodes = set([])
    data_list = []
    f = open(input_file, "r")
    for i, line in enumerate(f):
        if i % 3 == 0:  # time
            t.append(datetime.datetime.fromtimestamp(int(line)))
        elif i % 3 == 1:  # gpu id
            pass
        else:
            data = process_util_string(line)
            # print(list(data.keys()))
            data_nodes = data_nodes.union(set(data.keys()))
            data_list.append(data)
            # print(data.keys())
    # return t,a[:32],b[:8]
    data_nodes = list(data_nodes)
    data_nodes.sort()
    # print(data_nodes)
    # print(data_list[0])
    data_table = np.empty((len(data_nodes * n_gpu_per_node), len(t)))
    for i, tt in enumerate(t):
        one_time_step = np.array(
            combine_gpu_util(data_nodes, data_list[i], n_gpu_per_node)
        )
        data_table[:, i] = one_time_step
    return t, data_table, data_nodes


t, data, node_names = read_gpu_log()

fig = plot_gpu_utilization(
    t,
    data,  # shape (len(node_names) * n_gpu_per_node, len(time_steps))
    n_gpu_per_node=4,
    node_names=node_names,
    target_path="/data/html/palakons/vll-gpu-30-min-latest.png",
)
