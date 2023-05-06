import datetime
import os
import random
import subprocess
import time
from itertools import chain, combinations

import numpy as np
import pandas as pd
import requests
from pandas.io.json import json_normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale


# command1 = 'curl -s https://api.github.com/repos/ckolivas/lrzip/tags'
def get_version_info():
    res = requests.get('https://api.github.com/repos/ckolivas/lrzip/tags')
    data = res.json()
    json_list = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
    version_name = json_list['name'][0:15]  # only the version that can be install properly
    return version_name


def get_file_list(file_address):
    file_list = os.listdir(file_address)
    return file_list


def install_new_version(version_name):
    # lowest accept version is 0.610
    os.system('sudo rm -rf lrzip-*')
    os.system('sudo rm *.zip')
    os.system('sudo wget https://github.com/ckolivas/lrzip/archive/refs/tags/' + version_name + '.zip')
    time.sleep(5)
    os.system('sudo unzip ' + version_name + '.zip')
    time.sleep(5)
    os.chdir('lrzip-' + version_name[0:])
    os.system('sudo ./autogen.sh')
    os.system('sudo ./configure')
    os.system('sudo make -j `nproc`')
    os.system('sudo make install')


def catch_commit_between_version(version1, version2):
    os.chdir('lrzip')
    command = 'git log --pretty=oneline ' + version1 + '...' + version2
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
    sha_context = str(p).split("\n")
    sha = []
    for item in sha_context:
        if len(item) != 0:
            sha.append(item[0:40])
    os.chdir('../lrzipScript')
    return sha


def install_new_commit(commit_sha):
    os.system('sudo git checkout ' + commit_sha)
    k = subprocess.Popen('sudo ./autogen.sh', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    k = subprocess.Popen('sudo ./configure', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    # print(k)
    k = subprocess.Popen('sudo make -j `nproc`', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    k = subprocess.Popen('sudo make install', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]


def calculate_time(command_type, file_address, file):
    command = 'sudo ' + command_type + file_address + file
    print(command)
    record = []
    for i in range(3):
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE).communicate()[0]
        print(p)
        time0 = p.split("\n")[-2]
        time1 = time0[12:]
        x = time.strptime(time1.split('.')[0], '%H:%M:%S')
        sec_time = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
        sec_time += float(time1.split('.')[1]) / 100
        record.append(sec_time)
        os.system('sudo rm ' + file_address + file + '.lrz')
    avg_time = sum(record) / len(record)
    return avg_time


def transfer_real_data(real_data, old_pred, sampled_id):
    mix_result = []
    old_pred = np.asarray(old_pred)
    for i in range(0, len(real_data)):
        if i not in sampled_id:
            mix_result.append(old_pred[i])
        else:
            mix_result.append(real_data[i])
    return mix_result


def pick_up_command(weight_list, command_list):
    rand_number = random.randint(0, sum(weight_list))
    temp = 0
    i = 0
    while temp < rand_number:
        temp += weight_list[i]
        ++i
    return command_list[i - 1]


def generate_config_lrzip():
    config_option = []
    all_possible_configs = []
    for algorithm in ['-z', '-b', '-g', '-n', '-l']:
        for level in range(8, 10):
            for window in range(1, 100, 20):
                for nice in range(-20, 20, 8):
                    for processor in range(1, 5):
                        _cmd = 'sudo lrzip {} -L {} -w {} -N {} -p {}'.format(algorithm, level, window, nice, processor)
                        all_possible_configs.append([algorithm, level, window, nice, processor])
                        config_option.append(_cmd)
    return config_option, all_possible_configs


def transfer_config_lrzip(all_possible_configs):
    fea_algo_feature = np.eye(5)
    fea_algo_list = ['-b', '-g', '-l', '-n', '-z']
    all_possible_configs_cur = all_possible_configs
    all_possible_configs_cur = np.asanyarray(all_possible_configs_cur)
    config_features = []
    for possible_config in all_possible_configs_cur:
        algo_feature = fea_algo_feature[fea_algo_list.index(possible_config[0])]
        config_features.append(np.concatenate([algo_feature, np.asarray(possible_config[1:], dtype=float)]))
    config_features = np.asarray(config_features)
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(config_features)
    config_features = scaler.transform(config_features)
    return config_features


def powerset(list_name):
    s = list(list_name)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_config_llvm():
    config_option = []
    all_possible_configs = []
    configs = ['--gvn ', '--instcombine ', '--inline ', '--jump-threading ', '--simplifycfg ', '--sccp ', '--ipsccp ', '--licm ',
               '--iv-users ']
    all_combination = powerset(configs)
    for x in all_combination:
        if len(x) > 0:
            cmd = ''
            for item in x:
                cmd += item
            config_option.append(cmd)
    for a1 in range(0, 2):
        for a2 in range(0, 2):
            for a3 in range(0, 2):
                for a4 in range(0, 2):
                    for a5 in range(0, 2):
                        for a6 in range(0, 2):
                            for a7 in range(0, 2):
                                for a8 in range(0, 2):
                                    for a9 in range(0, 2):
                                        all_possible_configs.append([a1, a2, a3, a4, a5, a6, a7, a8, a9])
    all_possible_configs.remove([0]*9)
    return config_option, all_possible_configs


def transfer_config_llvm(all_possible_configs):
    config_features = np.asarray(all_possible_configs)
    return config_features


def generate_config_x264():
    asm = ['', '--no-asm ']
    x8dct = ['', '--no-8x8dct ']
    cabac = ['', '--no-cabac ']
    deblock = ['', '--no-deblock ']
    pskip = ['', '--no-fast-pskip ']
    mbtree = ['', '--no-mbtree ']
    mixed_refs = ['', '--no-mixed-refs ']
    weightb = ['', '--no-weightb ']
    rc_lookahead = ['--rc-lookahead 20 ', '--rc-lookahead 40 ']
    rc_value = [20, 40]
    ref = ['--ref 1 ', '--ref 5 ', '--ref 9 ']
    ref_value = [1, 5, 9]
    eye_2 = np.eye(2)
    config_option = []
    all_possible_configs = []
    for element0 in asm:
        for element1 in x8dct:
            for element2 in cabac:
                for element3 in deblock:
                    for element4 in pskip:
                        for element5 in mbtree:
                            for element6 in mixed_refs:
                                for element7 in weightb:
                                    for element8 in rc_lookahead:
                                        for element9 in ref:
                                            _cmd = '../x264/x264 '
                                            _cmd = _cmd + element0
                                            _cmd = _cmd + element1
                                            _cmd = _cmd + element2
                                            _cmd = _cmd + element3
                                            _cmd = _cmd + element4
                                            _cmd = _cmd + element5
                                            _cmd = _cmd + element6
                                            _cmd = _cmd + element7
                                            _cmd = _cmd + element8
                                            _cmd = _cmd + element9
                                            config_option.append(_cmd)
                                            v0 = eye_2[asm.index(element0)]
                                            v1 = eye_2[x8dct.index(element1)]
                                            v2 = eye_2[cabac.index(element2)]
                                            v3 = eye_2[deblock.index(element3)]
                                            v4 = eye_2[pskip.index(element4)]
                                            v5 = eye_2[mbtree.index(element5)]
                                            v6 = eye_2[mixed_refs.index(element6)]
                                            v7 = eye_2[weightb.index(element7)]
                                            v8 = float(rc_value[rc_lookahead.index(element8)])
                                            v9 = float(ref_value[ref.index(element9)])
                                            all_possible_config = np.concatenate((v0, v1, v2, v3, v4, v5, v6, v7, v8, v9), axis=None)
                                            all_possible_configs.append(all_possible_config)
                                            # print(_cmd)
                                            config_option.append(_cmd)
    return config_option, all_possible_configs


def transfer_config_x264(all_possible_configs):
    config_features = all_possible_configs
    config_features = np.asarray(config_features)
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(config_features)
    config_features = scaler.transform(config_features)
    return config_features


def generate_config_sqlite():
    config_option = []
    all_possible_configs = []
    cache_size_list = [' -DSQLITE_DEFAULT_CACHE_SIZE=4000', ' -DSQLITE_DEFAULT_CACHE_SIZE=2000']
    auto_index_list = [' -DSQLITE_OMIT_AUTOMATIC_INDEX', '']
    page_size_list = [' -DSQLITE_DEFAULT_PAGE_SIZE=512', ' -DSQLITE_DEFAULT_PAGE_SIZE=1024', ' -DSQLITE_DEFAULT_PAGE_SIZE=2048']
    locking_mode_list = [' -DSQLITE_DEFAULT_LOCKING_MODE=0', ' -DSQLITE_DEFAULT_LOCKING_MODE=1']
    omit_feature_list = [' -DSQLITE_OMIT_AUTOMATIC_INDEX', ' -DSQLITE_OMIT_BTREECOUNT',
                                         ' -DSQLITE_OMIT_BETWEEN_OPTIMIZATION', ' -DSQLITE_OMIT_LIKE_OPTIMIZATION',
                                         ' -DSQLITE_OMIT_LOOKASIDE', ' -DSQLITE_OMIT_QUICKBALANCE', ' -DSQLITE_OMIT_OR_OPTIMIZATION',
                                         ' -DSQLITE_OMIT_SHARED_CACHE', ' -DSQLITE_OMIT_XFER_OPT']
    store_type_list = [' -DSQLITE_TEMP_STORE=0', ' -DSQLITE_TEMP_STORE=1', ' -DSQLITE_TEMP_STORE=2', ' -DSQLITE_TEMP_STORE=3']
    disable_feature_list = [' -DSQLITE_DISABLE_LFS', ' -DSQLITE_DISABLE_DIRSYNC']
    autovacuum_list = [' -DSQLITE_DEFAULT_AUTOVACUUM=0', ' -DSQLITE_DEFAULT_AUTOVACUUM=1']
    eye_2 = np.eye(2)
    eye_3 = np.eye(3)
    eye_4 = np.eye(4)
    eye_9 = np.eye(9)
    for cache_size in cache_size_list:
        for auto_index in auto_index_list:
            for page_size in page_size_list:
                for locking_mode in locking_mode_list:
                    for omit_feature in omit_feature_list:
                        for store_type in store_type_list:
                            for disable_feature in disable_feature_list:
                                for autovacuum in autovacuum_list:
                                    cmd = '-DSQLITE_ENABLE_RTREE -DSQLITE_ENABLE_MEMSYS5 ' + cache_size
                                    cmd += auto_index
                                    cmd += page_size
                                    cmd += locking_mode
                                    cmd += omit_feature
                                    cmd += store_type
                                    cmd += disable_feature
                                    cmd += autovacuum
                                    v0 = eye_2[cache_size_list.index(cache_size)]
                                    v1 = eye_2[auto_index_list.index(auto_index)]
                                    v2 = eye_3[page_size_list.index(page_size)]
                                    v3 = eye_2[locking_mode_list.index(locking_mode)]
                                    v4 = eye_9[omit_feature_list.index(omit_feature)]
                                    v5 = eye_4[store_type_list.index(store_type)]
                                    v6 = eye_2[disable_feature_list.index(disable_feature)]
                                    v7 = eye_2[autovacuum_list.index(autovacuum)]
                                    all_possible_config = np.concatenate((v0, v1, v2, v3, v4, v5, v6, v7), axis=None)
                                    all_possible_configs.append(all_possible_config)
                                    config_option.append(cmd)
    return config_option, all_possible_configs


def transfer_config_sqlite(all_possible_configs):
    config_features = np.asarray(all_possible_configs)
    return config_features

