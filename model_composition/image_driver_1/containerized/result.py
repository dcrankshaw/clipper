""" Parse the results in dir gpu_and_batch_size_experiments/"""
import glob
import os
import json
import numpy as np

files = glob.glob("./gpu_and_batch_size_experiments/*.json")
results = dict()

for f in files:
    with open(f) as data_file:    
        data = json.load(data_file)

    thrus = np.mean(data["client_metrics"]["thrus"][1:])
    p99_lats = np.mean(data["client_metrics"]["p99_lats"][1:])
    mean_lats = np.mean(data["client_metrics"]["mean_lats"][1:])

    model = data["node_configs"][0]["name"]
    cpus_per_replica = data["node_configs"][0]["cpus_per_replica"]
    batch_size = data["node_configs"][0]["batch_size"]
    num_gpus = len(data["node_configs"][0]["gpus"])

    temp = data["clipper_metrics"]["histograms"]
    app_p99 = temp[1]["app:" + model + ":prediction_latency"]["p99"]
    model_p99 = temp[2]["model:" + model + ":1:prediction_latency"]["p99"]

    if model not in results:
        fname = "./gpu_and_batch_size_experiments/" + model + ".txt"
        results[model] = fname
        res = open(fname, "w")
        res.close()
    else:
        fname = results[model]

    res = open(fname, "a")
    res.write("num_gpus: {}, num_cpus: {}, batch_size: {}, mean_lats: {}, p99_lats: {}, thrus: {}, model_p99: {}, app_p99: {}\n".format(
        num_gpus, cpus_per_replica, batch_size, mean_lats, p99_lats, thrus, model_p99, app_p99))

