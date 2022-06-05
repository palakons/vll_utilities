import csv
import datetime
import pprint
import re
from os import initgroups

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
    print("im here 4")
    for i in range(len(axs)):
        ax = axs[i]
        print("axis", i)
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
    print("imhere 5")
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
    how_many_days_ago=None,
):
    print("open", input_file)
    t = []

    data_nodes = set([])
    data_list = []
    user_list = []
    flops_list = []
    f = open(input_file, "r")
    start_line = -5
    data, users, flops = None, None, None
    is_failed = False
    tod = datetime.datetime.now()
    for i, line in enumerate(f):
        if (
            len(line.split()) > 1
            and len(line.split()[-1]) == 10
            and line.split()[-1].isnumeric()
        ):  # if it is the date/time (in unix epoch format) line
            # print('problem',i,line)
            start_line = i
            line = line.split()[-1]
        try:  # if convertable to int, then start line here
            int(line)
            # start line 1
            start_line = i
            # print('startline',start_line,line)
        except:
            pass
        if i == start_line:  # time, if this is the "time" line
            try:  # convert to datetime format
                t_data = datetime.datetime.fromtimestamp(int(line))
            except:
                is_failed = True
                print(f"error time: line {i} content {line}")
        elif i == start_line + 1 and (
            how_many_days_ago is None
            or t_data > tod - datetime.timedelta(days=how_many_days_ago)
        ):  # if it's one line after date/time, and still in display range (e.g. one month before today)
            # the gpu id line
            if line.find("GPU") == -1:
                print("start_line+1:  # gpu id error", i, line)
            try:
                flops = process_gpuid_string(line, tflops_list)
                # print('flops',flops) #{'v01': [{'gpuid': 'GPU-ss', 'flop': 14.2}, {'gpuid': 'GPU-dd, 'flop': 14.2}, {'gpuid': 'GPU-ff', 'flop': 14.2}, {'gpuid': 'GPU-hh', 'flop': 14.2}], }
            except:
                is_failed = True
                print(f"error gpu: {i} ")

            # pass
        elif i == start_line + 2 and (
            how_many_days_ago is None
            or t_data > tod - datetime.timedelta(days=how_many_days_ago)
        ):  # if it's two lines after date/time (the user line)
            if line.find("GPU") != -1:  # errir if thet is "GPU" in this line!
                print("start_line+2:  # util string error", i, line)
            try:
                data, users = process_util_string_with_user(line)
                # print('data',data) #data {'v01': [0, 29, 15, 0], }
                # print('users',users)#users {'v01': ['mint', 'mint'], }
                data_nodes = data_nodes.union(set(data.keys()))  # list ofpossible nodes

                if not is_failed:
                    data_list.append(data)
                    user_list.append(users)
                    flops_list.append(flops)
                    t.append(t_data)

                    diff = set.difference(set(data.keys()), set(flops.keys()))
                    if len(diff) > 0:
                        print(
                            t_data,
                            ":read_gpu_log_2:there are some missing GPUID/flops data",
                            diff,
                        )
                else:
                    print(
                        "not add",
                    )
            except:
                print(f"error util: {i} ")
            # print(list(data.keys()))
            is_failed = False

    data_nodes = list(data_nodes)
    data_nodes.sort()  # to update to handle both "v01" and "v1"

    return t, data_list, data_nodes, user_list, flops_list


def process_util_string_with_user(line):
    # still need to handle two users in one gpu
    data = {}
    users = {}
    # print(line)

    pattern = r"Cluster v[0-9]{1,2} Cluster "
    mod_string = re.sub(pattern, " ", line)
    pattern = r" \| "
    mod_string = re.sub(pattern, " ", mod_string)
    # print(mod_string)
    token = mod_string.split()
    # print("token", token)
    current_token = ""
    for tag in token:
        if tag[0] == "v":
            current_token = tag
            if len(current_token) == 2:
                current_token = "v0" + current_token[-1:]
        elif tag[0].isdigit():  # number
            if current_token not in data:
                data[current_token] = [int(tag)]
            else:
                data[current_token].append(int(tag))
        else:  # name
            if tag != "":
                if current_token not in users:
                    users[current_token] = [tag]
                else:
                    users[current_token].append(tag)
            else:
                print("empty tag")

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
    pattern = r"v[0-9a-fA-F]{1,2}: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
    mod_string = re.sub(pattern, "", line)

    pattern = r"Unable to determine the device handle for gpu 0000:[0-9a-fA-F]{2}:00.0: Unknown Error"
    mod_string = re.sub(pattern, "", mod_string)

    pattern = r"Unable to determine the device handle for gpu 0000:[0-9a-fA-F]{2}:00.0: GPU is lost.  Reboot the system to recover this GPU"
    mod_string = re.sub(pattern, "", mod_string)

    pattern = (
        r"v[0-9a-fA-F]{1,2}: Failed to initialize NVML: Driver/library version mismatch"
    )
    mod_string = re.sub(pattern, "", mod_string)

    pattern = r"v[0-9a-fA-F]{1,2}: No devices found."
    mod_string = re.sub(pattern, "", mod_string)

    token = mod_string.split()
    # print(token)
    current_token = ""
    id_not_found = []
    for tag in token:
        if tag[0] == "v":
            current_token = tag[:-1]
            if len(current_token) == 2:
                current_token = "v0" + current_token[-1:]
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
        for gpuid in id_not_found:
            if gpuid[:3] != "GPU":
                raise
    return data


def pad_zero_util(gpu_util_list, n_gpu_per_node=4, pad_value=0):
    if len(gpu_util_list) < n_gpu_per_node:
        return gpu_util_list + [
            pad_value for i in range(n_gpu_per_node - len(gpu_util_list))
        ]
    return gpu_util_list[:n_gpu_per_node]


def combine_gpu_util(node_names, data, n_gpu_per_node, pad_value=0):
    result = []
    for name in node_names:
        try:
            result += pad_zero_util(
                data[name] if name in data else [], n_gpu_per_node, pad_value=pad_value
            )
        except:
            print(f"error combine_gpu_util {name} {data[name]} ")
    return result


def data_to_table(t, data_list, data_nodes, n_gpu_per_node, pad_value=0):

    data_table = np.empty((len(data_nodes * n_gpu_per_node), len(t)))
    for i, tt in enumerate(t):
        try:
            one_time_step = np.array(
                combine_gpu_util(
                    data_nodes, data_list[i], n_gpu_per_node, pad_value=pad_value
                )
            )
            data_table[:, i] = one_time_step
        except:
            # pass
            print(f"error {i} {tt} {data_list[i]} {data_nodes}")
    return t, data_table, data_nodes


def remove_unmatched_utils(users, utils_org, output_tflops):
    utils = utils_org.copy()
    need_remove = len(utils) - len(users)
    cache = 0
    for i in range(need_remove):
        v_to_remove = 0 if output_tflops else 1
        if v_to_remove not in utils:  # error from GPUtil, split the biggest
            v_to_remove = max(utils)
            cache += v_to_remove
            print("something wrong", f"remove {v_to_remove}, cahce {cache}")
        utils.remove(v_to_remove)
    for i in range(len(utils)):
        utils[i] += cache / len(utils)
    return utils


def process_user_util_zero(users, utils_org, flops, output_tflops=True):
    user_util = {}
    utils = utils_org.copy()
    # print("ut flp",flops,utils,)
    if len(flops) != len(utils):
        print("flops", len(flops), len(utils))
        print("flops", flops, (utils))

    # print(utils)
    for i in range(len(flops)):
        if output_tflops:  # convert to tflops
            utils[i] *= flops[i]["flop"] / 100
        else:  # output count
            utils[i] = 1
    # print(utils)

    # aaa=  users.copy()
    # bbb = utils.copy()
    if len(users) < len(utils):
        utils = remove_unmatched_utils(users, utils, output_tflops)

    #     need_remove = len(utils) - len(users)
    #     cache = 0
    #     for i in range(need_remove):
    #         v_to_remove = 0 if output_tflops else 1
    #         if v_to_remove not in utils: # error from GPUtil, split the biggest
    #             v_to_remove = max(utils)
    #             cache += v_to_remove
    #             print(
    #                 "something wrong",f'remove {v_to_remove}, cahce {cache}'
    #             )
    #         utils.remove(v_to_remove)
    #     for i in range(len(utils)):
    #         utils[i] += cache / len(utils)

    # # print("compare",utils,utils2)
    # l1=utils.copy()
    # l1.sort()
    # l2=utils2
    # l2.sort()
    # if(l1!=l2):
    #     return "Non equal"

    if len(users) < len(utils):
        print("why")
    # print(len(users) ,'len(users) < len(utils)',len(utils),users,utils)

    # print('user_util')
    # pprint.pprint(utils)

    for i, user in enumerate(users):
        # print("user",user)
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
    # print('data',data) #data {'v01': [0, 29, 15, 0], }
    data_nodes,
    user_list,
    # print('users',users)#users {'v01': ['mint', 'mint'], }
    flops_list,
    # print('flops',flops) #{'v01': [{'gpuid': 'GPU-ss', 'flop': 14.2}, {'gpuid': 'GPU-dd, 'flop': 14.2}, {'gpuid': 'GPU-ff', 'flop': 14.2}, {'gpuid': 'GPU-hh', 'flop': 14.2}], }
    time_min_max=None,
    is_counting_whole_gpu=False,
):
    if time_min_max is None:
        # time_range = [len(t)-1,len(t)-1]
        time_range = list(range(len(t))[-100:-5])
    else:
        time_range = range(time_min_max[0], time_min_max[1])

    util_by_user_per_time = []

    for i in time_range:  # each time step
        total_utilization = {}
        total_utilization_per_time = {}
        total_utilization_per_node = {}
        time_diff = time_diff_from_idx(t, [i])
        for node in data_nodes:  # each node: v01, etc.
            if node in user_list[i]:  # if vxx si online at time i
                if node in flops_list[i]:
                    # print(t[time_range[i]])
                    # print('lensss',len(user_list[i][node]),len(data_list[i][node]),len(flops_list[i][node]))
                    util_dict = None
                    # print(data_list[i].keys())
                    if len(data_list[i][node]) != len(flops_list[i][node]):
                        util_dict = {}
                    else:
                        util_dict = process_user_util_zero(
                            user_list[i][node],
                            data_list[i][node],
                            flops_list[i][node],
                            output_tflops=not is_counting_whole_gpu,
                        )
                    # print("util_dict",util_dict)
                    # {'penguin': 4}
                    # pprint.pprint(util_dict)
                    for user in util_dict:
                        comma_split_users = user.split(",")
                        for (
                            user_extract
                        ) in (
                            comma_split_users
                        ):  # split utilization equally when multiple job on the same gpu
                            # if user_extract =='':
                            #     print('empty user_extract',total_utilization.keys())
                            gpu_util_share = (
                                util_dict[user] / len(comma_split_users) * time_diff
                            )
                            if user_extract in total_utilization:
                                total_utilization[user_extract] += gpu_util_share
                            else:
                                total_utilization[user_extract] = gpu_util_share

                            if user_extract in total_utilization_per_time:
                                total_utilization_per_time[
                                    user_extract
                                ] += gpu_util_share
                            elif user_extract != "":
                                total_utilization_per_time[
                                    user_extract
                                ] = gpu_util_share
                else:
                    print(
                        t[i],
                        "utlization_by_users,cannot find flops",
                        node,
                        "at time",
                    )
                    # print("data_list", data_list[i].keys())
                    # print("flops_list", flops_list[i].keys())
                    # print(user_list[i])
                    # print(flops_list[i])

        for user in total_utilization_per_time:
            total_utilization_per_time[user] /= time_diff
            if user == "":
                print("total_utilization_per_time", total_utilization_per_time)
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
        if len(name) < 1:
            results.append("")
        else:
            results.append(name[0] + "*" * (len(name) - 2) + name[-1])
    return results


def util_by_user_to_table(utlization):
    all_names = sorted(list(set([b for a in utlization for b in list(a.keys())])))
    print("all names", all_names)
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


def table_to_csv(t, data_table, data_nodes, n_gpu_per_node=4, outfile=None):
    print("output", outfile)

    data_nodes = sorted(data_nodes)

    header = ["time", "node_gpu", "value", "user"]

    # print(header)
    with open(outfile, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(header)
        # print(len(t),len(data_table[0]))
        for i in range(len(t)):
            # print([t[i]],data_table[i])
            nodes = [
                node + f"g{i:02d}" for node in data_nodes for i in range(n_gpu_per_node)
            ]
            for i_node in range(len(nodes)):
                spamwriter.writerow(list([t[i], nodes[i_node], data_table[i_node][i]]))


def divide_gpu_per_user(data):
    storage = {}

    for users in data:

        gpu_value = data[users]
        comma_split_users = users.split(",")
        # print('comma_split_users',comma_split_users)
        for (
            user_extract
        ) in (
            comma_split_users
        ):  # split utilization equally when multiple job on the same gpu
            gpu_util_share = gpu_value / len(comma_split_users)
            if user_extract in storage:
                storage[user_extract] += gpu_util_share
            else:
                storage[user_extract] = gpu_util_share
    return storage


def process_gpu_per_node(
    data_list, user_list, data_nodes, tflops_list, output_tflops=True
):
    result = []
    for time_idx in range(len(data_list)):  # one tiem step
        result_per_time = {}

        for node in data_nodes:  # eahc node
            if (
                node in user_list[time_idx] 
            ):  # if node is being used
                if node in tflops_list[time_idx]:
                    # data = data_list[time_idx][node]
                    # if len(data_list[time_idx][node]) != len(user_list[time_idx][node]):
                    #     # pprint.pprint(data_list[time_idx][node])
                    #     # pprint.pprint(user_list[time_idx][node])
                    #     data = remove_unmatched_utils(
                    #         user_list[time_idx][node],
                    #         data_list[time_idx][node],
                    #         output_tflops,
                    #     )
                    # # pprint.pprint(tflops_list)
                    # print("data b4", user_list[time_idx][node], data_list[time_idx][node])
                    if len(data_list[time_idx][node]) != len(tflops_list[time_idx][node]):
                        data = {}
                    else:
                        data = process_user_util_zero(
                            user_list[time_idx][node],
                            data_list[time_idx][node],
                            tflops_list[time_idx][node],
                            output_tflops=output_tflops,
                        )
                    # print('data',data)
                    # pprint.pprint(data) #summarized TFlops per comma, user per node
                    storage = divide_gpu_per_user(data)
                    # print('data')
                    # pprint.pprint(data) #summarized TFlops per individual user per node
                    result_per_time[node] = storage
                else:
                    print(
                        time_idx,
                        "process_gpu_per_node: cannot find flops",
                        node,
                    )
                    # print(user_list[i])
                    # print(flops_list[i])
        result.append(result_per_time)

    return result


def per_node_to_csv_long(tt, data, outfile=None):
    print("output: per_node_to_csv_long", outfile)
    if len(tt) == len(data):

        header = ["time", "node", "user", "all_user"]
        with open(outfile, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(header)
            for time_idx in range(len(tt)):
                for node in data[time_idx]:
                    if len(data[time_idx][node]) > 0:
                        max_user = max(
                            data[time_idx][node], key=data[time_idx][node].get
                        )
                        spamwriter.writerow(
                            [tt[time_idx], node, max_user, data[time_idx][node]]
                        )
    else:
        print("no csv, diff data")
