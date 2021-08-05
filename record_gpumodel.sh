export tl_clusters="v01,v02,v03,v04,v05,v06,v07,v08,v09,v10,v11,v12,v13,v14,v15,v16,v17"
for i in $(echo $tl_clusters | sed "s/,/ /g")
do
    # call your procedure/other scripts here below
    printf  "$i: " >> /home2/palakons/track_cluster_usage.log
    ssh $i nvidia-smi -L  >> /home2/palakons/track_gpu_model.log
done 