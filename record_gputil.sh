date +%s >> /home2/palakons/track_cluster_usage.log
# export tl_clusters="v01,v02,v03,v04,v05,v06,v07,v08,v09,v10,v11,v12,v13,v14,v15,v16,v17"
export tl_clusters="v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23"
for i in $(echo $tl_clusters | sed "s/,/ /g")
do
    # call your procedure/other scripts here below
    printf  "$i: " >> /home2/palakons/track_cluster_usage.log
    ssh $i nvidia-smi -L | sed 's/GPU.*UUID: //g' |tr '\n' ' ' | sed 's/) / /g' >> /home2/palakons/track_cluster_usage.log
done 
printf "\n" >> /home2/palakons/track_cluster_usage.log
python ~/cluster_utils/tasklauncher.py lsgpu | sed 's/\x1b\[[0-9;]*m//g' |tr '\n' ' ' | sed 's/Cluster v[0-9]. Cluster //g' >> /home2/palakons/track_cluster_usage.log
printf "\n" >> /home2/palakons/track_cluster_usage.log