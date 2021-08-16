from shared_functions import *

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

gpu_whole =  True
t, data_list, data_nodes, user_list, flops_list = read_gpu_log_2(
    tflops_list, input_file="test.log"
)
# print(t, data_list)#,
# print(data_nodes)#
# print(user_list)
# print(flops_list)

tt, n_gpu_online, total_tflops, util_by_user_per_time = utlization_by_users(
    t,
    data_list,
    data_nodes,
    user_list,
    flops_list,
    time_min_max=[len(t) * 0, len(t)],
    is_counting_whole_gpu=gpu_whole,
)


# print(tt)
# print(n_gpu_online)
# print(total_tflops)
# print(util_by_user_per_time)
# print(len(tt),len(n_gpu_online),len(total_tflops),len(util_by_user_per_time))
# print(n_gpu_online,total_tflops,util_by_user_per_time)

# plot_gpu_utilization_per_user(tt,n_gpu_online,total_tflops,util_by_user_per_time,save_location="/home/palakons/vll_utilities/per_users.png",is_counting_whole_gpu=gpu_whole)

if False:
    t, data_table, data_nodes = data_to_table(
        t, data_list, data_nodes, n_gpu_per_node=4
    )  # output flops
    fig = plot_gpu_utilization(
        t,
        data_table,  # shape (len(node_names) * n_gpu_per_node, len(time_steps))
        n_gpu_per_node=4,
        node_names=data_nodes,
        # target_path="/data/html/palakons/vll-gpu-30-min-latest.png",
        target_path="/home/palakons/vll_utilities/vll-gpu-30-min-latest_test.png",
    )

if True:
    t, data_table, data_nodes = data_to_table(
        t, data_list, data_nodes, n_gpu_per_node=4, pad_value=None
    )  # output flops
    # table_to_csv(
    #     t,
    #     data_table,
    #     data_nodes,
    #     n_gpu_per_node=4,
    #     outfile="/data/html/palakons/track_heatmap.csv",
    # )


# array_to_csv(tt, user_list,n_gpu_online, total_tflops, util_by_user_per_time,outfile='/data/html/palakons/track_gpu_whole.csv')

gpu_whole = False
t, data_list, data_nodes, user_list, flops_list = read_gpu_log_2(tflops_list)
# tt, n_gpu_online, total_tflops, util_by_user_per_time = utlization_by_users(t, data_list, data_nodes, user_list, flops_list,time_min_max=[len(t)*0,len(t)],is_counting_whole_gpu=gpu_whole)
# array_to_csv(tt, user_list,n_gpu_online, total_tflops, util_by_user_per_time,outfile='/data/html/palakons/track_tflops.csv')


# print("data_list")
# pprint.pprint(data_list)
# print("user_list")
# pprint.pprint(user_list)


main_user_per_node = process_gpu_per_node(data_list, user_list, data_nodes)
per_node_to_csv_long( t,main_user_per_node, outfile="/data/html/palakons/track_main_user_per_node.csv")