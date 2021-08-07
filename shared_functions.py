import datetime
from os import initgroups
import pprint
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

    fig, axs = plt.subplots(
        n_node * n_gpu_per_node, sharex=True, sharey=True, figsize=(10, 10)
    )
    plt.xticks(rotation=15)
    axs[0].set_title(title)
    my_cmap = plt.get_cmap("RdYlGn_r")
    rescale = lambda y: (y - 0) / (np.max(gpu_util) - 0)

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


def read_gpu_log_2(
    tflops_list,
    input_file="/home2/palakons/track_cluster_usage.log",
    n_gpu_per_node=4,
):

    t = []

    data_nodes = set([])
    data_list = []
    user_list = []
    flops_list = []
    f = open(input_file, "r")
    for i, line in enumerate(f):
        if i % 3 == 0:  # time
            try:
                t.append(datetime.datetime.fromtimestamp(int(line)))
            except:
                print(f"error: {i} {line}")
        elif i % 3 == 1:  # gpu id
            # print(line)
            flops = process_gpuid_string(line, tflops_list)
            flops_list.append(flops)
            # pass
        else:

            data, users = process_util_string_with_user(line)
            # print(list(data.keys()))
            data_nodes = data_nodes.union(set(data.keys()))
            data_list.append(data)
            user_list.append(users)
    data_nodes = list(data_nodes)
    data_nodes.sort()

    return t, data_list, data_nodes, user_list, flops_list


def process_util_string_with_user(line):
    # still need to handle two users in one gpu
    data = {}
    users = {}
    # print(line)
    pattern = r" \| "
    mod_string = re.sub(pattern, " ", line)
    # print(mod_string)
    token = mod_string.split()
    # print(token)
    current_token = ""
    for tag in token:
        if tag[0] == "v":
            current_token = tag
        elif tag[0].isdigit():  # number
            if current_token not in data:
                data[current_token] = [int(tag)]
            else:
                data[current_token].append(int(tag))
        else:  # name
            if current_token not in users:
                users[current_token] = [tag]
            else:
                users[current_token].append(tag)

    # print(data, users)
    return data, users


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


def process_gpuid_string(line, tflops_list):
    data = {}
    pattern = r"v[0-9]{2}: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
    mod_string = re.sub(pattern, "", line)

    token = mod_string.split()
    # print(token)
    current_token = ""
    id_not_found = []
    for tag in token:
        if tag[0] == "v":
            current_token = tag[:-1]
        else:
            flop = 0
            try:
                flop = tflops_list[tag]
            except:
                id_not_found.append(tag)
                pass
            item = {"gpuid": tag, "flop": flop}
            if current_token not in data:
                data[current_token] = [item]
            else:
                data[current_token].append(item)
    # print(data)
    if len(id_not_found) > 0:
        print(f"error: flops not found: {id_not_found}")
    return data


def pad_zero_util(gpu_util_list, n_gpu_per_node=4):
    if len(gpu_util_list) < n_gpu_per_node:
        return gpu_util_list + [0] * (n_gpu_per_node - len(gpu_util_list))
    return gpu_util_list[:n_gpu_per_node]


def combine_gpu_util(node_names, data, n_gpu_per_node):
    result = []
    for name in node_names:
        try:
            result += pad_zero_util(data[name] if name in data else [], n_gpu_per_node)
        except:
            print(f"error combine_gpu_util {name} {data[name]} ")
    return result


def data_to_table(t, data_list, data_nodes, n_gpu_per_node):

    data_table = np.empty((len(data_nodes * n_gpu_per_node), len(t)))
    for i, tt in enumerate(t):
        try:
            one_time_step = np.array(
                combine_gpu_util(data_nodes, data_list[i], n_gpu_per_node)
            )
            data_table[:, i] = one_time_step
        except:
            # pass
            print(f"error {i} {tt} {data_list[i]} {data_nodes}")
    return t, data_table, data_nodes


def process_user_util_zero(users, utils, flops):
    user_util = {}
    utils_org = utils.copy()
    if len(flops) != len(utils):
        print("flops", len(flops), len(utils))
        print("flops", flops, (utils))

    # print(utils)
    for i in range(len(flops)):
        utils[i] *= flops[i]["flop"] / 100
    # print(utils)

    if len(users) < len(utils):
        need_remove = len(utils) - len(users)
        for i in range(need_remove):
            utils.remove(0)

    if len(users) < len(utils):
        print("why")

    for i, user in enumerate(users):
        if user not in user_util:
            user_util[user] = utils[i]
        else:
            user_util[user] += utils[i]
    # print(user_util)
    # if user_util[list(user_util.keys())[0]] == 0 and sum(utils_org)>0:
    #     print(user_util.keys(),users,utils_org)
    return user_util


def time_diff_from_idx(t, time_range):
    if time_range[0] != 0:
        time_diff = t[time_range[-1]] - t[time_range[0] - 1]
    else:
        time_diff = t[time_range[-1]] - (t[0] - (t[1] - t[0]))

    return time_diff.total_seconds()


def utlization_by_users(
    t,
    data_list,
    data_nodes,
    user_list,
    flops_list,
    time_min_max=None,
    is_counting_whole_gpu=False,
):
    total_utilization = {}
    if time_min_max is None:
        # time_range = [len(t)-1,len(t)-1]
        time_range = list(range(len(t))[-100:-5])
    else:
        time_range = range(time_min_max[0], time_min_max[1])

    util_by_user_per_time = []
    for i in time_range:  # each time step
        total_utilization_per_time = {}
        time_diff = time_diff_from_idx(t, [i])
        for node in data_nodes:  # each node: v01, etc.
            if node in user_list[i]:
                util_dict = process_user_util_zero(
                    user_list[i][node], data_list[i][node], flops_list[i][node]
                )
                for user in util_dict:
                    comma_split_users = user.split(",")
                    for (
                        user_extract
                    ) in (
                        comma_split_users
                    ):  # split utilization equally when multiple job on the same gpu
                        gpu_util_share = (
                            (1 if is_counting_whole_gpu else util_dict[user])
                            / len(comma_split_users)
                            * time_diff
                        )
                        if user_extract in total_utilization:
                            total_utilization[user_extract] += gpu_util_share
                        else:
                            total_utilization[user_extract] = gpu_util_share

                        if user_extract in total_utilization_per_time:
                            total_utilization_per_time[user_extract] += gpu_util_share
                        else:
                            total_utilization_per_time[user_extract] = gpu_util_share

        for user in total_utilization_per_time:
            total_utilization_per_time[user] /= time_diff
        util_by_user_per_time.append(total_utilization_per_time)
    print(
        f"since start tracking, folow users in the tracking: {len(total_utilization.keys())}"
    )

    time_diff = time_diff_from_idx(t, time_range)
    for user in total_utilization:
        total_utilization[user] /= time_diff

    print(
        f"Average flops-hour from {t[time_range[0]]} to {t[time_range[-1]]}, {time_diff} seconds"
    )
    pprint.pprint(total_utilization)

    sum_tflops = 0
    # i = len(t) - 1
    gpu_count = set([])
    n_gpu_online = []
    total_tflops = []
    for i in time_range:
        gpu_count_per_time = set([])
        sum_tflops_per_time = 0
        time_diff = time_diff_from_idx(t, [i])
        for node in data_nodes:
            if node in flops_list[i]:
                for item in flops_list[i][node]:
                    gpu_count = gpu_count.union([item["gpuid"]])
                    gpu_count_per_time = gpu_count_per_time.union([item["gpuid"]])
                    sum_tflops += item["flop"] * time_diff
                    sum_tflops_per_time += item["flop"]
        n_gpu_online.append(len(gpu_count_per_time))
        total_tflops.append(sum_tflops_per_time)

    print(f"average utilzed {sum(total_utilization.values()):.2f}")
    if is_counting_whole_gpu:
        print(f"average {sum(n_gpu_online)/len(n_gpu_online):.2f} availble gpu")
        print(
            f" {sum(total_utilization.values())/(sum(n_gpu_online)/len(n_gpu_online))*100:.2f}% utilization"
        )
    else:
        time_diff = time_diff_from_idx(t, time_range)
        sum_tflops /= time_diff
        print(f"average {sum_tflops:.2f} availble tera flops")
        print(f" {sum(total_utilization.values())/(sum_tflops)*100:.2f}% utilization")
    print(f" on different {len(gpu_count)} GPU IDs")

    return (
        t[time_range[0] : time_range[-1] + 1],
        n_gpu_online,
        total_tflops,
        util_by_user_per_time,
    )


def cloak_names(names):
    results = []
    for name in names:
        results.append(name[0] + "*" * (len(name) - 2) + name[-1])
    return results


def util_by_user_to_table(utlization):
    all_names = sorted(list(set([b for a in utlization for b in list(a.keys())])))

    output = dict((k, []) for k in all_names)
    for util in utlization:
        data = dict.fromkeys(all_names, 0)

        for user in util:
            data[user] = util[user]

        for user in data:
            output[user].append(data[user])
    return output


def plot_gpu_utilization_per_user(
    tt,
    n_gpu_online,
    total_tflops,
    util_by_user_per_time,
    save_location=None,
    is_counting_whole_gpu=False,
):
    fig = plt.figure()
    title = "GPU utilization by user: " + (
        "counting GPU whole" if is_counting_whole_gpu else "couting tflops"
    )
    plt.title(title)
    if is_counting_whole_gpu:
        plt.plot(tt, n_gpu_online, label="gpu online")
        plt.ylabel("#GPUs")
    else:
        plt.plot(tt, total_tflops, label="available tflops")
        plt.ylabel("tflops")
    # plt.plot(tt,[sum(a.values()) for a in util_by_user_per_time],label = 'utilized tflops')
    stacked_data = util_by_user_to_table(util_by_user_per_time)
    # print(stacked_data.values())
    plt.stackplot(tt, stacked_data.values(), labels=cloak_names(stacked_data.keys()))

    plt.legend(bbox_to_anchor=(1, -0.15), ncol=3)
    plt.xlabel("time")
    plt.tight_layout()
    if save_location is not None:
        fig.savefig(save_location)
    return fig


def array_to_csv_long(
    tt, n_gpu_online, total_tflops, util_by_user_per_time, outfile=None
):
    print("output", outfile)
    import csv

    print(util_by_user_per_time[0])

    header = ["time", "user", "value"]
    with open(outfile, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header)
        for i in range(len(tt)):
            for user in util_by_user_per_time[i]:
                spamwriter.writerow([tt[i], user, util_by_user_per_time[i][user]])


def array_to_csv(
    tt, user_list, n_gpu_online, total_tflops, util_by_user_per_time, outfile=None
):
    print("output", outfile)
    import csv

    users = sorted(
        list(set([user for k in util_by_user_per_time for user in list(k.keys())]))
    )

    header = ["time"] + users
    # print(header)
    with open(outfile, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header)
        for i in range(len(tt)):
            spamwriter.writerow(
                [tt[i]]
                + [
                    util_by_user_per_time[i][user]
                    if user in util_by_user_per_time[i]
                    else 0
                    for user in users
                ]
            )
