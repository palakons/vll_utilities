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
    "GPU-1752fe77-b429-57bd-710d-838b77262e94": 14.2,
    "GPU-c16084b6-a103-083b-27af-dcb754a03ba2": 14.2,
    "GPU-d3978bc0-7f73-2078-3fe8-4e17861f7940": 14.2,
    "GPU-1bf76776-5945-6bb0-0033-84f4abaffbaa": 14.2,
    "GPU-93875c93-99c8-a4bd-04ad-ab495feb1382": 14.2,
    "GPU-73892c65-11ab-5167-e741-011efd6659f0": 14.2,
    "GPU-3b27ec93-b71d-6a65-6cdc-04be0077dfbd": 14.2,
    "GPU-26ed4582-9975-d3df-f3ba-c51f61c9bfd8": 14.2,
    "GPU-3a6450a4-da15-6490-a440-09a803184ab8": 14.2,
    "GPU-559fb878-260b-1a52-a0e8-1df706c75b73": 14.2,
    "GPU-8fe7d644-812b-31bb-902e-d183108bc3f2": 14.2,
    "GPU-1f720ddc-493f-3ce2-e13e-0068f6958ec7": 14.2,
    "GPU-7789ab1e-18de-e06b-3c24-6add1648ab9a": 14.2,
    "GPU-d9baa4a3-b6c7-5e82-01b7-94188336f9fa": 14.2,
    "GPU-fc2f96c8-0125-9573-6b7a-b01f1bf75e33": 14.2,
    "GPU-19b1bc18-bd4e-16f5-e063-3c1471297762": 14.2,  # 2080Ti https://irendering.net/what-are-teraflops-in-rtx-2080-ti/#:~:text=Nvidia's%20GeForce%20RTX%202080%20Ti,an%20impressive%20sign%20of%20growth.
    "GPU-544fceb6-3ac8-bcc5-d3e7-ca1e5c797a96": 19.17,  # A4000 https://www.techpowerup.com/gpu-specs/rtx-a4000.c3756
    "GPU-e81377be-dc65-b031-ef37-b6ad2c4bab90": 19.17,
    "GPU-21191f4d-426f-0709-66f8-85c72ef3da5c": 19.17,
    "GPU-a5811ce1-3bc3-baa0-4c85-e406011f6598": 19.17,
    "GPU-f685352a-507f-3087-9805-c89ca83403a2": 14.2,
    "GPU-39ed36f1-f26d-a0d0-ac7f-1d61a7a92f0c": 36,
    "GPU-62752c0c-d2f1-835f-5e3b-6a80d162914b": 36,
    "GPU-23e9bb8d-0191-0256-9efb-0087e42a95b7": 36,
    "GPU-e24ca2dc-4a04-6963-34e9-e0fb712d4790": 36,
    "GPU-5eafeb54-8429-7d72-cbac-b8177cc623e0": 36,
    "GPU-971dd3c0-ebf3-4601-0902-e22622d513de": 36,
    "GPU-e46a9552-0385-4593-1ec3-7734bb353438": 36,
    "GPU-b1053fa5-5227-4134-29fd-3f8c08aaa263": 36,
    "GPU-20f97ca4-232b-cd52-5385-c45abab9bdad": 36,
    "GPU-845a7345-cfee-d136-c413-38a7c5841e59": 14.2,
    "GPU-389c8a87-e2eb-4cc0-c9dd-a08a98cae8bd": 14.2,
    "GPU-adb4f661-68d9-fbc5-c90d-9a7fe28e1165": 19.17,
    "GPU-bfffff39-e2da-c87a-5a45-115da1764e3b": 19.17,
    "GPU-81c27bf1-9baf-ba3f-aa78-673dabce72fe": 19.17,
    "GPU-a1dd7619-e9bc-62a6-4e17-a94f0aaf1787": 14.2,
    "GPU-32b3d712-965c-1bff-c5b0-ad9d71eae77e": 19.17,
    "GPU-1c8afcd2-19a0-a9a2-2794-817ebd4f5972": 19.17,
    "GPU-e52c4d41-6124-0781-4adf-93013b7d2bbc": 19.17,
    "GPU-b653cc78-4374-bf47-d49b-f52562db3be3": 14.2,
    "GPU-e835c0f5-eb5d-4a36-c95d-55756d4cd4d7": 19.17,
    
}

gpu_whole = False
t, data_list, data_nodes, user_list, flops_list = read_gpu_log_2(
    tflops_list, how_many_days_ago=30
)
print("im'here", len(t))
# print(data_nodes)

t, data_table, data_nodes = data_to_table(
    t, data_list, data_nodes, n_gpu_per_node=4
)  # output flops
print("im'here2")
# fig = plot_gpu_utilization(
#     t,
#     data_table,  # shape (len(node_names) * n_gpu_per_node, len(time_steps))
#     n_gpu_per_node=4,
#     node_names=data_nodes,
#     target_path="/data/html/palakons/vll-gpu-30-min-latest.png",
# )
# print("im'here3")

tt, n_gpu_online, total_tflops, util_by_user_per_time = utlization_by_users(
    t, data_list, data_nodes, user_list, flops_list, time_min_max=[len(t) * 0, len(t)]
)
# print('tt',tt)

array_to_csv(
    tt,
    user_list,
    n_gpu_online,
    total_tflops,
    util_by_user_per_time,
    outfile="/data/html/palakons/track_tflops.csv",
)

# plot_gpu_utilization_per_user(
#     tt,
#     n_gpu_online,
#     total_tflops,
#     util_by_user_per_time,
#     save_location="/data/html/palakons/vll-gpu-user-latest.png",
# )
main_user_per_node = process_gpu_per_node(
    data_list, user_list, data_nodes, flops_list, output_tflops=not gpu_whole
)
per_node_to_csv_long(
    t, main_user_per_node, outfile="/data/html/palakons/track_main_user_per_node.csv"
)

gpu_whole = True
t, data_list, data_nodes, user_list, flops_list = read_gpu_log_2(
    tflops_list, how_many_days_ago=30
)

t, data_table, data_nodes = data_to_table(
    t, data_list, data_nodes, n_gpu_per_node=4, pad_value=None
)  # output flops
table_to_csv(
    t,
    data_table,
    data_nodes,
    n_gpu_per_node=4,
    outfile="/data/html/palakons/track_heatmap.csv",
)


tt, n_gpu_online, total_tflops, util_by_user_per_time = utlization_by_users(
    t,
    data_list,
    data_nodes,
    user_list,
    flops_list,
    time_min_max=[len(t) * 0, len(t)],
    is_counting_whole_gpu=gpu_whole,
)

array_to_csv(
    tt,
    user_list,
    n_gpu_online,
    total_tflops,
    util_by_user_per_time,
    outfile="/data/html/palakons/track_gpu_whole.csv",
)
# print(len(tt),len(n_gpu_online),len(total_tflops),len(util_by_user_per_time))
# print(n_gpu_online,total_tflops,util_by_user_per_time)

plot_gpu_utilization_per_user(
    tt,
    n_gpu_online,
    total_tflops,
    util_by_user_per_time,
    save_location="/data/html/palakons/vll-gpu-whole-latest.png",
    is_counting_whole_gpu=gpu_whole,
)
