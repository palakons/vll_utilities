import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size


tflops_list = {
    "GPU-709743fc-7c2a-81ce-0c04-14d30db6edfe": 14.2,
    "GPU-fc38f444-f7a5-342f-6afd-cdce4537670f": 14.2,
    "GPU-8f87452d-8ca3-62f8-08fd-bbb639426bdf": 14.2,
    "GPU-0ac1380f-e140-ee31-0b0f-2991a1191a27": 14.2,
    "GPU-1d301f66-da89-6170-00bc-1448333501bf": 14.2,
    "GPU-981c38b6-be76-33ce-b474-db9bf366adb0": 14.2,
    "GPU-861da7f2-2a19-1a8d-710d-dbec979994bb": 14.2,
    "GPU-bde6bafe-9a1e-9933-d874-cd8facace9f5": 14.2,
    "GPU-d509e023-248e-370b-68c3-9091e17dac92": 14.2,
    "GPU-696dc756-820c-bd63-c584-242cfeff4c53": 14.2,
    "GPU-60655ed4-289a-3987-47dd-be8d29d11421": 14.2,
    "GPU-bd632a6d-5642-46d8-3ca1-fd38320c46d7": 14.2,
    "GPU-5389821d-af03-3611-1ce1-795cf59a8a77": 14.2,
    "GPU-590da2d7-8e0d-3729-e4e0-8d41b3330c55": 14.2,
    "GPU-ed7108de-30ea-1506-9785-e0cef62f2482": 14.2,
    "GPU-f057c390-17c2-32f5-5a06-8ecfd571284d": 14.2,
    "GPU-166f858e-40ae-0006-276a-d6dcdb23174f": 14.2,
    "GPU-53ee9a56-de55-747b-97ba-8dfb0345308e": 14.2,
    "GPU-1b1027c1-4493-50c6-847b-4bc6b8db7b4b": 14.2,
    "GPU-d226a05e-6ebb-47b2-f886-cf9b0e4204bc": 14.2,
    "GPU-d6427358-18de-7389-d749-9fee856a8a88": 14.2,
    "GPU-8154f12a-19ae-7f7b-3da9-83434b9b95be": 14.2,
    "GPU-5654b919-d9d4-61e6-bc55-da6d376dc6ed": 14.2,
    "GPU-241b6cc9-84f7-288d-ecd6-86d8a8ec704b": 14.2,
    "GPU-801d4dcb-b4ff-aecd-fa7c-065306e2d913": 14.2,
    "GPU-c906383f-7e86-0651-2d27-9f6245448864": 14.2,
    "GPU-fcc3eaff-2360-13a2-d63a-77e5df784724": 14.2,
    "GPU-88ff1b5a-3107-2139-f12c-2ddad2b0eef8": 14.2,
    "GPU-608582fa-dcfd-ec05-faaa-3a616f3addb3": 14.2,
    "GPU-776d0b27-5f56-3752-7d5b-4dc8b8f703e1": 14.2,
    "GPU-61a88860-d3a8-d820-1667-16e08d223618": 14.2,
    "GPU-fa88bbf4-8215-2239-3cc8-4ca58971b85d": 14.2,
    "GPU-5328c663-e065-ed74-b5d9-ffea9a574d7c": 14.2,
    "GPU-85e88f14-c7ae-c9d8-73f5-f947feaba51d": 14.2,
    "GPU-7e9a99c4-faee-e1cb-0225-87df6061afc3": 14.2,
    "GPU-1b167847-7424-b181-a321-3923e4b75704": 14.2,
    "GPU-610b3862-ebd7-42c3-5bab-55368ffe6392": 14.2,
    "GPU-f1597fe1-10f1-5ff8-e239-15df54deb5c4": 14.2,
    "GPU-1983e0ba-d893-c31a-36c1-0b044598b8c2": 14.2,
    "GPU-2fedffb3-fbb1-c583-c741-7867812a88da": 14.2,
    "GPU-8c244175-3223-5878-b832-e1cbaa97f8da": 14.2,
    "GPU-da80facd-6867-80ef-da30-78e5b2a9968a": 14.2,
    "GPU-f9533bed-feaa-f66e-aeb9-31bf9a032cc1": 14.2,
    "GPU-521ff969-2665-0c95-d7ae-aae37e4c1b7e": 14.2,
    "GPU-c0bc88af-5077-9fe7-c699-4ab041312954": 14.2,
    "GPU-ab3e32cf-9c94-e85b-ff40-cb9a2b1eb725": 14.2,
    "GPU-52861426-4537-abc1-0d13-53c27a144152": 36,
    "GPU-81fa636f-e5f3-07c4-2462-f9a343ee662e": 36,
}

# https://www.digitaltrends.com/computing/nvidia-rtx-3090-vs-rtx-2080-ti-most-powerful-gaming-gpus-duke-it-out/


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


def process_gpuid_string(line):
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


def read_gpu_log_2(
    input_file="/home2/palakons/track_cluster_usage.log",
    n_gpu_per_node=4,
):

    t = []

    data_nodes = set([])
    data_list = []
    user_list=[]
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
            flops = process_gpuid_string(line)
            flops_list.append(flops)
            # pass
        else:

            data, users = process_util_string_with_user(line)
            # print(list(data.keys()))
            data_nodes = data_nodes.union(set(data.keys()))
            data_list.append(data)
            user_list.append(users)
            # print(data.keys())
    # return t,a[:32],b[:8]
    data_nodes = list(data_nodes)
    data_nodes.sort()
    # print(data_nodes)
    # print(data_list[0])

    return t, data_list, data_nodes, user_list, flops_list


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


def read_gpu_log(
    input_file="/home2/palakons/track_cluster_usage.log",
    n_gpu_per_node=4,
):

    t = []

    data_nodes = set([])
    data_list = []
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
            flops = process_gpuid_string(line)
            flops_list.append(flops)
            # pass
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
        try:
            one_time_step = np.array(
                combine_gpu_util(data_nodes, data_list[i], n_gpu_per_node)
            )
            data_table[:, i] = one_time_step
        except:
            pass
            print(f"error {i} {tt} {data_list[i]} {data_nodes}")
    return t, data_table, data_nodes

def process_user_util_zero(users,utils,flops):
    user_util = {}
    utils_org  = utils.copy()
    if len(flops) != len(utils):
        print("flops", len(flops),len(utils))
    
    # print(utils)
    for i in range(len(flops)):
        utils[i] *=flops[i]['flop']/100
    # print(utils)

    if  len(users) < len(utils):
        need_remove = len(utils)-len(users) 
        for i in range(need_remove):
            utils.remove(0)
    
    if  len(users) < len(utils):
        print('why')
    
    for i,user in enumerate(users):
        if user not in user_util:
            user_util[user] = utils[i]
        else:
            user_util[user]  +=utils[i]
    # print(user_util)
    # if user_util[list(user_util.keys())[0]] == 0 and sum(utils_org)>0:
    #     print(user_util.keys(),users,utils_org)
    return user_util


def utlization_by_users(t, data_list, data_nodes, user_list, flops_list):
    # print(t[0])
    # print( data_list[0])
    # print(data_nodes[0])
    # print( user_list[0].items())
    # print( flops_list[0])
    sum_tflops = 0
    # for i,tt in enumerate(t):
    i = len(t)-1
    gpu_count = 0
    for node in data_nodes:
        # print(flops_list[i][node])
        if node in flops_list[i]:
            for item in flops_list[i][node]:
                gpu_count+=1
                sum_tflops += item['flop']
    print(f"Latest: total terra flops {sum_tflops}")
    print(f" on {gpu_count} active GPUs")

    total_utilization = {}
    for i,tt in enumerate(t):
        for node in data_nodes:
            if node in user_list[i]:
                # print(len(user_list[i][node]),len(data_list[i][node]))
                util_dict = process_user_util_zero(user_list[i][node],data_list[i][node],flops_list[i][node])
                # if  len(user_list[i][node]) < len(data_list[i][node]):
                #     print(user_list[i][node],data_list[i][node])
                #     print()
                for k in util_dict:
                    if k in total_utilization:
                        total_utilization[k] += util_dict[k]
                    else:
                        total_utilization[k] = util_dict[k]
    print(f"since start tracking, folow users in the tracking: {total_utilization.keys()}")
    print(total_utilization)





t, data_list, data_nodes, user_list, flops_list = read_gpu_log_2()
utlization_by_users(t, data_list, data_nodes, user_list, flops_list)

# t, data, node_names = data_to_table(t, data_list, data_nodes, n_gpu_per_node=4)

# line="v01: GPU-8f87452d-8ca3-62f8-08fd-bbb639426bdf GPU-0ac1380f-e140-ee31-0b0f-2991a1191a27 GPU-1d301f66-da89-6170-00bc-1448333501bf GPU-981c38b6-be76-33ce-b474-db9bf366adb0 v02: GPU-861da7f2-2a19-1a8d-710d-dbec979994bb GPU-bde6bafe-9a1e-9933-d874-cd8facace9f5 GPU-d509e023-248e-370b-68c3-9091e17dac92 GPU-696dc756-820c-bd63-c584-242cfeff4c53 v03: GPU-60655ed4-289a-3987-47dd-be8d29d11421 GPU-bd632a6d-5642-46d8-3ca1-fd38320c46d7 GPU-5389821d-af03-3611-1ce1-795cf59a8a77 GPU-590da2d7-8e0d-3729-e4e0-8d41b3330c55 v04: GPU-ed7108de-30ea-1506-9785-e0cef62f2482 GPU-f057c390-17c2-32f5-5a06-8ecfd571284d GPU-166f858e-40ae-0006-276a-d6dcdb23174f GPU-53ee9a56-de55-747b-97ba-8dfb0345308e v05: GPU-1b1027c1-4493-50c6-847b-4bc6b8db7b4b GPU-d226a05e-6ebb-47b2-f886-cf9b0e4204bc GPU-d6427358-18de-7389-d749-9fee856a8a88 GPU-8154f12a-19ae-7f7b-3da9-83434b9b95be v06: GPU-5654b919-d9d4-61e6-bc55-da6d376dc6ed GPU-241b6cc9-84f7-288d-ecd6-86d8a8ec704b GPU-801d4dcb-b4ff-aecd-fa7c-065306e2d913 GPU-c906383f-7e86-0651-2d27-9f6245448864 v07: GPU-fcc3eaff-2360-13a2-d63a-77e5df784724 GPU-88ff1b5a-3107-2139-f12c-2ddad2b0eef8 v08: GPU-608582fa-dcfd-ec05-faaa-3a616f3addb3 GPU-776d0b27-5f56-3752-7d5b-4dc8b8f703e1 GPU-61a88860-d3a8-d820-1667-16e08d223618 GPU-fa88bbf4-8215-2239-3cc8-4ca58971b85d v09: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.  v10: GPU-5328c663-e065-ed74-b5d9-ffea9a574d7c GPU-85e88f14-c7ae-c9d8-73f5-f947feaba51d v11: GPU-7e9a99c4-faee-e1cb-0225-87df6061afc3 GPU-1b167847-7424-b181-a321-3923e4b75704 v12: GPU-610b3862-ebd7-42c3-5bab-55368ffe6392 GPU-f1597fe1-10f1-5ff8-e239-15df54deb5c4 GPU-1983e0ba-d893-c31a-36c1-0b044598b8c2 v13: GPU-2fedffb3-fbb1-c583-c741-7867812a88da GPU-8c244175-3223-5878-b832-e1cbaa97f8da v14: GPU-da80facd-6867-80ef-da30-78e5b2a9968a GPU-f9533bed-feaa-f66e-aeb9-31bf9a032cc1 GPU-521ff969-2665-0c95-d7ae-aae37e4c1b7e GPU-c0bc88af-5077-9fe7-c699-4ab041312954 v15: GPU-ab3e32cf-9c94-e85b-ff40-cb9a2b1eb725 v16: v17: GPU-52861426-4537-abc1-0d13-53c27a144152 GPU-81fa636f-e5f3-07c4-2462-f9a343ee662e "
# process_gpuid_string(line)
# line ="v01  00 00 00 00   v02  86 85 100 87 penguin | penguin | penguin | penguin  v03  00 00 00 00   v04  00 00 00 00   v05  00 00 00 00   v06  89 00 92 85 supasorn | suttisak | suttisak  v07  00 00 suttisak | suttisak  v08  87 82 84 85 suttisak | suttisak | suttisak | suttisak  v09    v10  72 74 56 nontawat | nontawat | nontawat  v11    v12    v13    v14  00 00 00 00   v15  00 00   v16    v17  00 00   "
# data,users = process_util_string_with_user(line)

if False:
    fig = plot_gpu_utilization(
        t,
        data,  # shape (len(node_names) * n_gpu_per_node, len(time_steps))
        n_gpu_per_node=4,
        node_names=node_names,
        # target_path="/data/html/palakons/vll-gpu-30-min-latest.png",
        target_path="/home/palakons/vll_utilities/vll-gpu-30-min-latest_test.png",
    )
