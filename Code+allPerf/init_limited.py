import process
import process_limited
import pandas as pd
import numpy as np


def get_init_result(command_list, folder_address, file_list):
    store = []
    for command in command_list:
        f = file_list[0]  # just run one file in this occasion
        running_time_record = []
        for time in range(3):
            running_time = process.calculate_time(command, folder_address, f)
            running_time_record.append(running_time)
        avg_running_time = sum(running_time)/len(running_time)
    store.append([command, avg_running_time])
    running_time_df = pd.DataFrame(store, columns=["command", "time", "version", "file", "running_time"])
    running_time_df.to_csv("initial_time.csv")
    return running_time_df


def initial_version_install(version_list):
    process.install_new_version(version_list[-1])


def build_model(config, initial_idx, weight, all_data, start_point, stop_point, init_lenght):
    sampled_config_ids = list(np.random.randint(len(config), size=init_lenght))
    past_result = []
    weight, config_black_list, bias_point, stop_point, new_mix_result, sampled_config_ids = process_limited.comsa(config, initial_idx, weight, all_data
                                                                                                                  , start_point, sampled_config_ids, past_result,
                                                                                                                  stop_point)
    return weight, config_black_list, bias_point, stop_point, new_mix_result, sampled_config_ids


