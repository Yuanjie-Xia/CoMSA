import pandas as pd
import os
import subprocess
import time
import datetime
import requests
from pandas.io.json import json_normalize
import sys


def get_version_info():
    res = requests.get('https://api.github.com/repos/ckolivas/lrzip/tags')
    data = res.json()
    json_list = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
    version_name = json_list['name'][0:15]  # only the version that can be install properly
    return version_name


def generate_config():
    config_option = []
    for algorithm in ['-z', '-b', '-g', '-n', '-l']:
        for level in range(8, 10):
            for window in range(1, 100, 20):
                for nice in range(-20, 20, 8):
                    for processor in range(1, 5):
                        _cmd = 'sudo lrzip {} -L {} -w {} -N {} -p {}'.format(algorithm, level, window, nice, processor)
                        config_option.append(_cmd)
    return config_option


def main():
    # version_name = get_version_info()
    os.chdir('lrzip')
    command = 'sudo git log --pretty=oneline v0.650...v0.610'
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    sha_context = p.split('\n')
    sha = []
    for item in sha_context:
        if len(item) != 0:
            sha.append(item[0:40])

    os.chdir('')
    config_command = generate_config()
    num = int(sys.argv[1])
    item = sha[num]
    os.chdir('lrzip')
    print(item)
    os.system('sudo git checkout ' + item)
    k = subprocess.Popen('sudo ./autogen.sh', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    k = subprocess.Popen('sudo ./configure', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    # print(k)
    k = subprocess.Popen('sudo make -j `nproc`', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    k = subprocess.Popen('sudo make install', shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]
    os.chdir('')
    for i in range(1):
        df = []
        # directory = os.getcwd()
        # print(directory)
        for config in config_command:
            p = subprocess.Popen(config + ' test_files/linux-2.6.37.tar', shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE).communicate()[0]
            print(p)
            try:
                time0 = p.split("\n")[-2]
                time1 = time0[12:]
                print(time1)
                x = time.strptime(time1.split('.')[0], '%H:%M:%S')
                sec_time = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                              seconds=x.tm_sec).total_seconds()
                sec_time += float(time1.split('.')[1]) / 100
                os.system('sudo rm test_files/linux-2.6.37.tar.lrz')
                print([item, i, config, sec_time])
                df.append([item, i, config, sec_time])

            except:
                print("cannot install commit")
                os.system('sudo rm test_files/linux-2.6.37.tar.lrz')
                pass
        running_time_df = pd.DataFrame(df, columns=["commit", "time", "command", "running_time"])
        running_time_df.to_csv("commit_running_time" + str(num) + str(i) + ".csv")


if __name__ == "__main__":
    main()
