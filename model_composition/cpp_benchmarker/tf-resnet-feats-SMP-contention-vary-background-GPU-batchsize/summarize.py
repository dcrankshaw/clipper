import os
import sys
import json
import numpy as np


def summarize(fname):
    with open(fname, "r") as f:
        results = json.load(f)

    throughput_results = results["throughput_results"]
    summary = throughput_results["summary_metrics"]
    thrus = []
    lats = {
        "tf-resnet-feats": [],
        "tf-inception-contention": [],
        "tf-kernel-svm-contention": [],
    }
    for trial in summary:
        thrus.append(trial["client_thrus"]["tf-resnet-feats"])
        clipper_lats = trial["clipper_mean_lats"]
        for k in lats:
            lats[k].append(clipper_lats[k])
    # HERE
    mean_thru = np.mean(thrus)

    #HERE
    mean_lats = {}

    for k in lats:
        mean_lats[k] = np.mean(lats[k])
    kernel_measures = results["kernel_measures"]

    # HERE
    agg_km = {}

    for k in kernel_measures:
        agg_km[k] = np.mean(kernel_measures[k])

    # mean_user_ticks = np.mean(kernel_measures["user_ticks"])
    # mean_sys_ticks = np.mean(kernel_measures["sys_ticks"])
    # mean_wall_clock = np.mean(kernel_measures["wall_clock"])
    contention_configs = results["contention"]["contention_configs"]
    contention_gpu_batch = None
    contention_cpu_batch = None
    for c in contention_configs:
        if "inception" in c["name"]:
            contention_gpu_batch = c["batch_size"]
        if "kernel-svm" in c["name"]:
            contention_cpu_batch = c["batch_size"]
    print(("\nInception batch size: {ib}, KSVM batch size: {kb}\n"
            "Throughput: {thru}\n"
            "Container latencies: {lats}\n"
            "Kernel measures: {kern}\n\n").format(ib=contention_gpu_batch,
                                                kb=contention_cpu_batch,
                                                thru=mean_thru,
                                                lats=mean_lats,
                                                kern=agg_km))





def summarize_all():
    # fs = os.listdir(".")
    for f in sorted(os.listdir(".")):
        if f[-4:] == "json":
            summarize(f)
if __name__ == "__main__":
    summarize_all()
