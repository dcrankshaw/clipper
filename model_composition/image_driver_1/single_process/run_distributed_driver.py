import sys
import os
import json

from subprocess import Popen

CONFIG_KEY_BATCH_SIZE = "batch_size"
CONFIG_KEY_CPU_AFFINITIES = "cpu_affinities"
CONFIG_KEY_TAGGED_PROCESS_PATH = "tagged_process_path"
CONFIG_KEY_NUM_REPLICAS = "num_replicas"
CONFIG_KEY_TRIAL_LENGTH = "trial_length"
CONFIG_KEY_NUM_TRIALS = "num_trials"
CONFIG_KEY_SLO_MILLIS = "slo_millis"

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def get_gpus(replica_num):
    resnet_gpu = 2 * replica_num
    inception_gpu = (2 * replica_num) + 1
    return resnet_gpu, inception_gpu

def launch_processes(config):
    batch_size = config[CONFIG_KEY_BATCH_SIZE]
    cpu_affinities = config[CONFIG_KEY_CPU_AFFINITIES]
    process_path = config[CONFIG_KEY_TAGGED_PROCESS_PATH]
    num_replicas = int(config[CONFIG_KEY_NUM_REPLICAS])
    trial_length = config[CONFIG_KEY_TRIAL_LENGTH]
    num_trials = config[CONFIG_KEY_NUM_TRIALS]
    slo_millis = config[CONFIG_KEY_SLO_MILLIS]


    for replica_num in range(num_replicas): 
        resnet_gpu, inception_gpu = get_gpus(replica_num)

        cpu_affinity = cpu_affinities[replica_num]
        cpu_aff_list = cpu_affinity.split(" ")
        comma_delimited_cpu_aff = ",".join(cpu_aff_list)

        process_cmd = "(export CUDA_VISIBLE_DEVICES=\"{res_gpu},{incep_gpu}\";" \
                      " numactl -C {cd_cpu_aff} python driver.py -b {bs} -c {cpu_aff}" \
                      " -t {trials} -tl {length} -n {rep_num} -p {proc_file} -s {slo})".format(
                              res_gpu=resnet_gpu,
                              incep_gpu=inception_gpu,
                              cd_cpu_aff=comma_delimited_cpu_aff,
                              bs=batch_size,
                              cpu_aff=cpu_affinity,
                              trials=num_trials,
                              length=trial_length,
                              rep_num=replica_num,
                              proc_file=process_path,
                              slo=slo_millis)

        print("Running: {}".format(process_cmd))
        Popen(process_cmd, shell=True)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    launch_processes(config)
    
