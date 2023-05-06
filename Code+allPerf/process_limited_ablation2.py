import numpy as np
import pandas as pd
from modAL.models import ActiveLearner, CommitteeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import query


def transfer_real_data(real_data, old_pred, sampled_id):
    mix_result = []
    old_pred = np.asarray(old_pred)
    for i in range(0, len(real_data)):
        if i not in sampled_id:
            mix_result.append(old_pred[i])
        else:
            mix_result.append(real_data[i])
    return mix_result


def comsa(config_option, initial_idx, weight, all_data, count, sampled_config_ids, past_result_list,
                    commit_impact, config_impact, stop_point):
    # config_features_id = list(range(0, len(config_option)))
    # config_features = np.asarray(config_option, dtype=np.float64)
    sampled_config_ids_copy = sampled_config_ids.copy()
    result = all_data[all_data['commit_num'] == count]['time']
    if len(past_result_list) > 0:
        past_result = past_result_list[0]
    else:
        past_result = []
    config_features = np.asarray(config_option)
    #if len(past_result_list) > 0:
    #    mix_init_result = result.copy()
    #    np.random.seed(0)
    #    for i in range(len(sampled_config_ids)):
    #        mix_init_result[sampled_config_ids[i]] = past_result_list[0][sampled_config_ids[i]]
    #else:
    mix_init_result = result.copy()

    mix_init_result = np.asarray(mix_init_result)
    result = np.asarray(result)
    # x_train = [config_features[idx] for idx in initial_idx]
    # x_train = np.asarray(x_train)
    # print(x_train)
    # result_train = [result[idx] for idx in initial_idx]
    # result_train = np.asarray(result_train, dtype=np.float64)
    # print(result_train)
    learner_list = [ActiveLearner(
        estimator=XGBRegressor(),
        X_training=config_features[idx], y_training=mix_init_result[idx]
    ) for idx in initial_idx]
    # initializing the Committee
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=query.max_std_sampling
        #query_strategy=query.flexible_weight
    )
    model = XGBRegressor()
    n_queries = int(len(config_features) * 0.9)
    history_data = 0
    bias_point = []
    std_list = []
    loop_train_idx = []
    for idx in range(n_queries):
        # sampled_config_ids = list(dict.fromkeys(sampled_config_ids))
        X_train = config_features[sampled_config_ids_copy]
        # print(X_train)
        # X_train = bit_id_transfer(X_train,config_option)
        # X_train = [X_train]
        y_train = []
        for i in sampled_config_ids_copy:
            y_train.append(result[i])
        # y_train = [j for i,j in enumerate(result) if i in sampled_config_ids]
        y_train = np.asarray(y_train, dtype=np.float64)
        # print(y_train)
        # print(len(config_features))
        # print(sampled_config_ids)
        model.fit(X_train, y_train)

        if len(past_result) > 0:
            mix_result = []
            past_result = np.asarray(past_result)
            for i in range(0, len(result)):
                if i not in sampled_config_ids_copy:
                    mix_result.append(past_result[i])
                else:
                    mix_result.append(result[i])
            mix_error = []
            mix_normal_error = []
            for i in range(0, len(config_features)):
                error_element = (mix_result[i] - model.predict(config_features)[i]) / model.predict(config_features)[i]
                abs_error_element = abs(error_element)
                mix_error.append(abs_error_element)
                mix_normal_error.append(error_element)
        else:
            mix_result = result
            mix_error = []
            mix_normal_error = []
            for i in range(0, len(config_features)):
                error_element = (result[i] - model.predict(config_features)[i]) / model.predict(config_features)[i]
                abs_error_element = abs(error_element)
                mix_error.append(abs_error_element)
                mix_normal_error.append(error_element)

        loss_distance = sum(mix_error) / len(mix_error)
        print("loss distance: " + str(loss_distance))

        y_pred = model.predict(config_features)
        if idx > 0:
            std = np.std(mix_error)
            largest_error = max(mix_error)
            std_list.append(std)
            print("mix error std: " + str(std))
            print("mix error max: " + str(largest_error))
            #if r2_score(mix_result, y_pred) > 0.99 and (std < 0.05 and largest_error < 0.15):
            if idx >= stop_point:
                print("index: " + str(idx))
                print("score: " + str(r2_score(mix_result, y_pred)))
                break

        if history_data - np.std(mix_error) < 0:
            print(sampled_config_ids_copy[-1])
            bias_point.append(sampled_config_ids_copy[-1])
        history_data = np.std(mix_error)
        print("index: " + str(idx))
        print("score: " + str(r2_score(result, y_pred)))
        print("mix-result score: " + str(r2_score(mix_result, y_pred)))

        error = []
        max_std_query_idx, std = committee.query(config_features)
        print("before add weight: " + str(max_std_query_idx))
        weight = query.deploy_weight(weight, std, idx, file_blacklist=commit_impact, config_blacklist=config_impact)
        query_idx, query_instance = query.flexible_weight(weight, loop_train_idx)
        #query_idx = list(np.random.randint(len(config_features), size=1))
        print("query idx: " + str(query_idx))
        loop_train_idx.append(query_idx[0])
        sampled_config_ids_copy += list(query_idx)
        # query_idx_bit = bit_id_transfer(config_features[query_idx], config_option)

        # print(len(config_features[query_idx]))
        query_result = [j for i, j in enumerate(result) if i in query_idx]
        query_result = np.asarray(query_result, dtype=np.float64)
        committee.teach(config_features[query_idx], query_result)

    g_error = []
    pred_result = model.predict(config_features)
    for i in range(len(config_features)):
        config_imp = abs(mix_result[i] - pred_result[i]) / pred_result[i]
        g_error.append(config_imp)

    d4error = {'value': g_error}
    error_df = pd.DataFrame(d4error)
    error_df.to_csv('~/errorData/errorData' + str(count) + '.csv')

    for k in range(0, len(g_error)):
        g_error[k] = (g_error[k] - min(g_error)) / (max(g_error) - min(g_error))

    new_mix_result = transfer_real_data(result, pred_result, sampled_config_ids_copy)

    return weight, g_error, bias_point, idx, new_mix_result, sampled_config_ids_copy
