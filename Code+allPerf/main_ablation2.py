import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import init_limited
import process
import process_limited_ablation2


def calculate_distance(sampled_ids, config_set):
    selected_config = config_set[sampled_ids]
    selected_config = np.matrix(selected_config, dtype=np.float32)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(selected_config)
    return kmeans.labels_


def main(start, end):
    lrzip_config, config_signal = process.generate_config_lrzip()  # change subject by change generate_config_[SubjectName]
    all_input_signal = process.transfer_config_lrzip(config_signal)  # change subject by change transfer_config_[SubjectName]
    stop_point = 18  # modify stop point
    init_length = 6
    weight = [0] * len(all_input_signal)
    # np.random.seed(0)
    all_data = pd.read_csv("perf_data_final/perf_first61.csv", index_col=0)
    config = all_input_signal
    init_set = list(np.random.randint(len(all_input_signal), size=init_length))
    init_idx = np.array_split(init_set, 3)
    weight, config_blacklist, bias_point, stop_point, new_mix_result, sampled_config_ids = init_limited.build_model(config,
                                                                                                                    init_idx,
                                                                                                                    weight,
                                                                                                                    all_data,
                                                                                                                    start, stop_point,init_length)
    history_result = [new_mix_result]
    stop_point_set = [stop_point]
    # sampled_config_ids = init_set
    for count in range(end, end+1):
        sampled_config_ids = sampled_config_ids + bias_point
        sampled_config_ids = list(sorted(set(sampled_config_ids), key=sampled_config_ids.index))
        # sampled_config_ids = list(np.random.randint(len(all_input_signal), size=6))
        label_ids = calculate_distance(sampled_config_ids, all_input_signal)
        init_idx_0 = [sampled_config_ids[index] for index in range(len(sampled_config_ids)) if label_ids[index] == 0]
        init_idx_1 = [sampled_config_ids[index] for index in range(len(sampled_config_ids)) if label_ids[index] == 1]
        init_idx_2 = [sampled_config_ids[index] for index in range(len(sampled_config_ids)) if label_ids[index] == 2]
        init_idx = [init_idx_0, init_idx_1, init_idx_2]

        #init_idx = np.array_split(sampled_config_ids, 3)
        print(init_idx)
        #if len(sampled_config_ids) > 40:
        #    sampled_config_ids = sampled_config_ids[len(sampled_config_ids)-40:]
        print("count: " + str(count))
        print("sampled_config_ids: " + str(sampled_config_ids))
        weight, config_blacklist, bias_point, stop_point, new_mix_result, sampled_config_ids = process_limited_ablation2.comsa(
            config, init_idx, weight, all_data,
            count, sampled_config_ids, history_result, stop_point)

        history_result.append(new_mix_result)

        stop_point_set.append(stop_point)
    sp = {'stop point': stop_point_set}
    sp_df = pd.DataFrame(sp)
    sp_df.to_csv("~/sp.csv")


if __name__ == "__main__":
    for i in range(0, 60):
        main(i, i+1) # change start and end point to control history information that need to use
