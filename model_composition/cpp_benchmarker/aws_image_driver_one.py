import subprocess32 as subprocess
import os
import sys
import numpy as np
import time
import logging
import json
from clipper_admin import ClipperConnection, AWSContainerManager
from datetime import datetime
from containerized_utils import driver_utils
# import argparse
import hashlib

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "TIMEOUT"
# CLIPPER_ADDRESS = "localhost"

INCEPTION_FEATS = "inception"
TF_KERNEL_SVM = "tf-kernel-svm"
TF_LOG_REG = "tf-log-reg"
TF_RESNET = "tf-resnet-feats"

ALL_REMOTE_ADDRS = []

DIVERGENCE_THRESHOLD = 0.95


def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus,
                          remote_addr=None,
                          input_size=None):

    if model_name == INCEPTION_FEATS:
        image = "gcr.io/clipper-model-comp/inception-feats:bench"
        return driver_utils.HeavyNodeConfig(name=INCEPTION_FEATS,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_RESNET:
        image = "gcr.io/clipper-model-comp/tf-resnet-feats:bench"
        return driver_utils.HeavyNodeConfig(name=TF_RESNET,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_LOG_REG:
        image = "gcr.io/clipper-model-comp/tf-log-reg:bench"
        return driver_utils.HeavyNodeConfig(name=TF_LOG_REG,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_KERNEL_SVM:
        image = "gcr.io/clipper-model-comp/tf-kernel-svm:bench"
        return driver_utils.HeavyNodeConfig(name=TF_KERNEL_SVM,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)


def setup_clipper(addr_config_map):
    for addr, configs in addr_config_map.iteritems():
        cl = ClipperConnection(AWSContainerManager(host=addr, redis_port=6380))
        # cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        cl.stop_all(remote_addrs=ALL_REMOTE_ADDRS)
        cl.start_clipper(
            query_frontend_image="clipper/zmq_frontend:develop",
            redis_cpu_str="0",
            mgmt_cpu_str="0",
            query_cpu_str="0,16,1,17,2,18,3,19")
        time.sleep(10)
        # Collect all configs for same model into local and remote lists
        grouped_configs_local = {}
        grouped_configs_remotes = {}
        unique_nodes = set()
        for c in configs:
            if c.remote_addr is None:
                assert c.name not in grouped_configs_local
                grouped_configs_local[c.name] = c
            else:
                # If we've already encountered remote configs for this model
                # on different machines, append to existing remotes list.
                if c.name in grouped_configs_remotes:
                    grouped_configs_remotes[c.name].append(c)
                else:
                    grouped_configs_remotes[c.name] = [c]
            unique_nodes.add(c.name)
        for name in unique_nodes:
            local = None
            remotes = None
            if name in grouped_configs_local:
                local = grouped_configs_local[name]
            if name in grouped_configs_remotes:
                remotes = grouped_configs_remotes[name]
            assert remotes is None
            driver_utils.setup_heavy_node(cl, local, remotes, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")


def get_clipper_batch_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_batch_sizes = {}
    for h in hists:
        if "batch_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            mean = float(h[name]["mean"])
            mean_batch_sizes[model] = round(float(mean), 2)
    return mean_batch_sizes


def get_clipper_latencies(metrics_json):
    hists = metrics_json["histograms"]
    p99_latencies = {}
    mean_latencies = {}
    for h in hists:
        if "prediction_latency" in h.keys()[0] and "model" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            p99 = float(h[name]["p99"]) / 1000.0 / 1000.0
            mean = float(h[name]["mean"]) / 1000.0 / 1000.0
            p99_latencies[model] = p99
            mean_latencies[model] = mean
    return (mean_latencies, p99_latencies)


def get_clipper_queue_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_queue_sizes = {}
    for h in hists:
        if "queue_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = float(h[name]["mean"])
            mean_queue_sizes[model] = round(float(mean), 2)
    return mean_queue_sizes


def get_clipper_thruputs(metrics_json):
    meters = metrics_json["meters"]
    thrus = {}
    for m in meters:
        if "prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            rate = float(m[name]["rate"])
            thrus[name] = round(float(rate), 5)
    return thrus


def get_clipper_counts(metrics_json):
    counters = metrics_json["counters"]
    counts = {}
    for c in counters:
        if "internal" not in c.keys()[0]:
            name = c.keys()[0]
            count = c[name]["count"]
            counts[name] = int(count)
    return counts


def get_profiler_stats(cur_client_metrics):
    all_latencies = []
    all_thrus = []
    all_counts = []
    all_ingest_rates = []
    for metrics_json in cur_client_metrics:
        latencies = {}
        thrus = {}
        counts = {}
        ingest_rates = {}

        data_lists = metrics_json["data_lists"]
        for l in data_lists:
            if "prediction_latencies" in l.keys()[0]:
                name = l.keys()[0]
                model = name.split(":")[0]
                cur_lats = [float(list(i.values())[0]) for i in l[name]["items"]]
                # convert to seconds
                cur_lats = np.array(cur_lats) / 1000.0 / 1000.0
                latencies[model] = cur_lats

        meters = metrics_json["meters"]
        for m in meters:
            if "prediction_throughput" in m.keys()[0]:
                name = m.keys()[0]
                model = name.split(":")[0]
                rate = float(m[name]["rate"])
                thrus[model] = round(float(rate), 5)

        for m in meters:
            if "ingest_rate" in m.keys()[0]:
                name = m.keys()[0]
                model = name.split(":")[0]
                rate = float(m[name]["rate"])
                ingest_rates[model] = round(float(rate), 5)

        counters = metrics_json["counters"]
        counts = {}
        for c in counters:
            if "num_predictions" in c.keys()[0]:
                name = c.keys()[0]
                model = name.split(":")[0]
                count = c[name]["count"]
                counts[model] = int(count)
        all_latencies.append(latencies)
        all_thrus.append(thrus)
        all_ingest_rates.append(ingest_rates)
        all_counts.append(counts)
    agg_p99_latency = {}
    agg_mean_latency = {}
    for model in all_latencies[0]:
        lats = np.hstack([c[model] for c in all_latencies]).flatten()
        agg_p99_latency[model] = round(np.percentile(lats, 99), 3)
        agg_mean_latency[model] = round(np.mean(lats), 3)

    agg_thrus = {}
    for model in all_thrus[0]:
        total_thru = np.sum([c[model] for c in all_thrus])
        agg_thrus[model] = total_thru

    agg_ingest_rates = {}
    for model in all_ingest_rates[0]:
        total_ingest_rate = np.sum([c[model] for c in all_ingest_rates])
        agg_ingest_rates[model] = total_ingest_rate

    agg_counts = {}
    for model in all_counts[0]:
        total_count = np.sum([c[model] for c in all_counts])
        agg_counts[model] = total_count

    return (agg_mean_latency, agg_p99_latency, agg_thrus, agg_counts, agg_ingest_rates)


def load_clipper_metrics(clipper_path):
    with open(clipper_path, "r") as clipper_file:
        clipper_metrics_str = clipper_file.read().strip()
        if clipper_metrics_str[-1] == ",":
            clipper_metrics_str = clipper_metrics_str.rstrip(",")
            clipper_metrics_str += "]"
        try:
            clipper_metrics = json.loads(clipper_metrics_str)
        except ValueError:
            # logger.warn("Unable to parse clipper metrics: {}. Skipping...".format(e))
            return None
    return clipper_metrics


def print_clipper_metrics(clipper_metrics):
    results_dict = {}
    results_dict["batch_sizes"] = get_clipper_batch_sizes(clipper_metrics)
    results_dict["queue_sizes"] = get_clipper_queue_sizes(clipper_metrics)
    results_dict["clipper_thrus"] = get_clipper_thruputs(clipper_metrics)
    results_dict["clipper_counts"] = get_clipper_counts(clipper_metrics)
    results_dict["mean_lats"], results_dict["p99_lats"] = get_clipper_latencies(clipper_metrics)
    logger.info(("\nqueue sizes: {queue_sizes}"
                 "\nbatch sizes: {batch_sizes}"
                 "\nclipper p99 lats: {p99_lats}"
                 "\nclipper mean lats: {mean_lats}\n").format(**results_dict))


def print_stats(cur_client_metrics, trial_num):
    cur_client_metrics = [c[trial_num] for c in cur_client_metrics]
    results_dict = {}
    results_dict["client_mean_lats"], results_dict["client_p99_lats"], \
        results_dict["client_thrus"], results_dict["client_counts"], \
        results_dict["ingest_rates"] = get_profiler_stats(cur_client_metrics)
    logger.info(("\nThroughput: {client_thrus}\nP99 lat: {client_p99_lats}"
                "\nMean lat: {client_mean_lats}"
                 "\nIngest rates: {ingest_rates}\n").format(**results_dict))
    return results_dict


# def load_metrics(client_path, clipper_path):
def load_metrics(client_path, clipper_path=None):
    with open(client_path, "r") as client_file:

        client_metrics_str = client_file.read().strip()
        if client_metrics_str[-1] == ",":
            client_metrics_str = client_metrics_str.rstrip(",")
            client_metrics_str += "]"
        try:
            client_metrics = json.loads(client_metrics_str)
        except ValueError:
            # logger.warn("Unable to parse client metrics: {}. Skipping...".format(e))
            return None
    if clipper_path is not None:
        with open(clipper_path, "r") as clipper_file:
            clipper_metrics_str = clipper_file.read().strip()
            if clipper_metrics_str[-1] == ",":
                clipper_metrics_str = clipper_metrics_str.rstrip(",")
                clipper_metrics_str += "]"
            try:
                clipper_metrics = json.loads(clipper_metrics_str)
            except ValueError:
                # logger.warn("Unable to parse clipper metrics: {}. Skipping...".format(e))
                return None
        return client_metrics, clipper_metrics
    else:
        return client_metrics


def load_lineage(lineage_path):
    with open(lineage_path, "r") as f:
        parsed = [json.loads(l) for l in f]
    return parsed


# def tag_arrival_process(fname, num_clients):
#     with open(deltas_path, "r") as f:
#         deltas = np.array([float(l.strip()) for l in f]).flatten()

def get_arrival_proc_file(lam, cv):
    if cv == 1:
        arrival_file_name = "{lam}.deltas".format(lam=lam)
    else:
        arrival_file_name = "{lam}_{cv}.deltas".format(lam=lam, cv=cv)

    arrival_delay_file = os.path.join(("/home/ubuntu/plots-model-comp-paper/experiments/"
                                       "cached_arrival_processes/{f}").format(f=arrival_file_name))
    return arrival_delay_file


def get_arrival_proc_files(lam, cv, num_clients, out_dir):
    arrival_delay_file = get_arrival_proc_file(lam, cv)
    if num_clients == 1:
        return [arrival_delay_file]
    # else:
    #     div = get_arrival_proc_file(int(lam/num_clients), cv)
    #     return [div for _ in range(num_clients)]

    with open(arrival_delay_file, "r") as f:
        deltas = np.array([float(l.strip()) for l in f]).flatten()
    arrival_history = np.cumsum(deltas)

    # Assign each query to a specific client
    tags = np.array([np.random.randint(num_clients) for _ in range(len(arrival_history))])
    tagged_arrival_diffs = [np.diff(arrival_history[tags == i]) for i in range(num_clients)]
    deltas_base = os.path.basename(arrival_delay_file)
    out_files = []
    for i in range(num_clients):
        fname = os.path.join(out_dir, "CLIENT-{}-{}".format(i, deltas_base))
        out_files.append(fname)
        with open(fname, "w") as f:
            for d in tagged_arrival_diffs[i]:
                f.write("{}\n".format(d))

    return out_files




def run_e2e(addr_config_map, name_addr_map, trial_length, driver_path, profiler_cores_strs,
            lam, cv, num_clients, slo):
    assert len(addr_config_map) >= 1
    setup_clipper(addr_config_map)
    # clipper_address = CLIPPER_ADDRESS
    cls = [ClipperConnection(AWSContainerManager(host=addr, redis_port=6380)) for addr in addr_config_map]
    # cls = [ClipperConnection(DockerContainerManager(redis_port=6380)) for addr in addr_config_map]
    for cl in cls:
        cl.connect()

    time.sleep(30)
    log_dir = "/tmp/image_driver_one_profiler_logs_{ts:%y%m%d_%H%M%S}".format(ts=datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def run(target_throughput, num_trials, name, arrival_process, check_for_divergence):
        for cl in cls:
            cl.drain_queues()
        time.sleep(10)
        for cl in cls:
            cl.drain_queues()
        time.sleep(30)
        arrival_delay_files = get_arrival_proc_files(lam, cv, num_clients, log_dir)
        try:
            procs = {}
            for client_num in range(num_clients):
                log_path = os.path.join(log_dir, "{n}-{t}-{p}-{c}".format(n=name,
                                                                    t=target_throughput,
                                                                    p=arrival_process,
                                                                    c=client_num))
                cmd = ["numactl", "-C", profiler_cores_strs[client_num],
                       os.path.abspath(driver_path),
                       "--target_throughput={}".format(target_throughput),
                       "--request_distribution={}".format(arrival_process),
                       "--trial_length={}".format(trial_length),
                       "--num_trials={}".format(num_trials),
                       "--log_file={}".format(log_path),
                       "--clipper_address_resnet={}".format(name_addr_map[TF_RESNET]),
                       "--clipper_address_inception={}".format(name_addr_map[INCEPTION_FEATS]),
                       "--request_delay_file={}".format(arrival_delay_files[client_num]),
                       "--latency_budget_micros={}".format(int(slo * 1000 * 1000))]
                if client_num == 0:
                    cmd.append("--get_clipper_metrics")

                logger.info("Driver command: {}".format(" ".join(cmd)))
                client_path = "{p}-client_metrics.json".format(p=log_path)
                lineage_paths = {m: "{p}-{m}-query_lineage.txt".format(m=m, p=log_path)
                                for m in [TF_RESNET, INCEPTION_FEATS, TF_KERNEL_SVM, TF_LOG_REG]}
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                procs[client_num] = (proc, log_path, client_path, lineage_paths)
            clipper_paths = ["{p}-clipper_metrics_{addr}.json".format(
                p=log_path, addr=a) for a in addr_config_map]

            recorded_trials = 0
            summary_results = []
            done = False
            while recorded_trials < num_trials and not done:
                time.sleep(5)
                cur_client_metrics = []
                for client_num in procs:
                    proc, log_path, client_path, lineage_paths = procs[client_num]
                    if proc.poll() is not None:
                        if proc.returncode != 0:
                            logger.warn("Driver process {client} finished with return code {code}".format(
                                client=client_num, code=proc.returncode))
                            done = True
                            break
                    if not (os.path.exists(client_path)):
                        logger.info("metrics don't exist yet")
                        continue
                    client_metrics = load_metrics(client_path)
                    if client_metrics is not None:
                        cur_client_metrics.append(client_metrics)
                print_stats_this_iter = False
                if len(cur_client_metrics) == num_clients:
                    new_recorded_trials = min([len(c) for c in cur_client_metrics])
                    if new_recorded_trials > recorded_trials:
                        recorded_trials = new_recorded_trials
                        new_stats = print_stats(cur_client_metrics, new_recorded_trials - 1)
                        summary_results.append(new_stats)
                        print_stats_this_iter = True
                        # Run for a couple trials before checking for divergence
                        if check_for_divergence and recorded_trials > 2:
                            models_to_replicate = []
                            ingest_rates = new_stats["ingest_rates"]
                            for m, t in new_stats["client_thrus"].iteritems():
                                if m != "e2e":
                                    if t/ingest_rates[m] < DIVERGENCE_THRESHOLD:
                                        logger.info("GREPTHISBBBB: Detected divergence in model {m}: {t}/{i}".format(
                                            m=m, t=t, i=ingest_rates[m]))
                                        models_to_replicate.append(m)
                            # No need to keep running if we're already diverging
                            # if len(models_to_replicate) > 0:
                            #     # The "finally" block of the outer try statement will terminate
                            #     # the procs
                            #     # # break out of the while loop
                            #     # done = True
                            #     # # stop all the drivers
                            #     # for client_num in procs:
                            #     #     proc, _, _, _ = procs[client_num]
                            #     #     proc.terminate()
                            #     return None, models_to_replicate
                    # else:
                    #     logger.info("Recorded trials is still {}".format(new_recorded_trials))


                if print_stats_this_iter:
                    for clipper_path in clipper_paths:
                        if not (os.path.exists(clipper_path)):
                            logger.info("Clipper metrics don't exist yet")
                            continue
                        clipper_metrics = load_clipper_metrics(clipper_path)
                        if clipper_metrics and len(clipper_metrics) >= recorded_trials:
                            print_clipper_metrics(clipper_metrics[recorded_trials - 1])

            all_client_metrics = []
            for client_num in procs:
                proc, log_path, client_path, lineage_paths = procs[client_num]
                prof_stderr = proc.stderr.read().strip()
                if len(prof_stderr) > 0:
                    logger.info("stderr: {}".format(prof_stderr))
                try:
                    loaded_metrics = load_metrics(client_path)
                    lineages = {name: load_lineage(p) for name, p in lineage_paths.items()}
                    # lineages = None
                    if loaded_metrics is not None:
                        all_client_metrics.append(loaded_metrics)
                    else:
                        logger.error("Error loading final metrics")
                except ValueError as e:
                    logger.error("Unable to parse final metrics")
                    raise e
            return driver_utils.Results(all_client_metrics,
                                        None,
                                        summary_results,
                                        lineages), None
        except Exception as e:
            logger.exception(e)
        finally:
            for c in procs:
                procs[c][0].terminate()

    run(100, 7, "warmup", "constant", check_for_divergence=False)
    throughput_results, models_to_replicate = run(0, 12, "throughput", "file", check_for_divergence=True)

    for cl in cls:
        cl.stop_all(remote_addrs=ALL_REMOTE_ADDRS)
    return (throughput_results, models_to_replicate)


class BenchmarkConfigurationException(Exception):
    pass


def hash_file(fname):
    with open(fname, "rb") as f:
        fbytes = f.read()
        return hashlib.sha256(fbytes).hexdigest()


def assign_models_to_nodes(resnet_addr, inception_addr, config):
    """

    This method first assigns ResNet and Inception to nodes, then assigns
    the ksvm reps to the ResNet node and the logistic regression models to
    the Inception node.
    Parameters
    ----------
    addrs: list
        A list of available addresses for running models.

    """
    MAX_GPUS_PER_NODE = 4

    # Map from address to resource name
    addr_resource_map = {
        resnet_addr: {"cpus": list(range(4, 16)),
                      "gpus": list(range(MAX_GPUS_PER_NODE))
                      },
        inception_addr: {"cpus": list(range(4, 16)),
                         "gpus": list(range(MAX_GPUS_PER_NODE))
                         }
    }


    # First assign GPU models greedily. We know that Resnet will always have the most
    # replicas, so start by assigning those.
    node_configs = config["node_configs"]
    res_config = node_configs[TF_RESNET]
    addr_config_map = {resnet_addr: [], inception_addr: []}
    name_addr_map = {TF_RESNET: resnet_addr, INCEPTION_FEATS: inception_addr}
    try:
        num_local_res_replicas = min(res_config["num_replicas"], MAX_GPUS_PER_NODE)
        num_remote_res_replicas = max(res_config["num_replicas"] - MAX_GPUS_PER_NODE, 0)
        local_res_node_conf = get_heavy_node_config(
            model_name=TF_RESNET,
            batch_size=int(res_config["batch_size"]),
            num_replicas=num_local_res_replicas,
            cpus_per_replica=res_config["num_cpus"],
            allocated_cpus=[addr_resource_map[name_addr_map[TF_RESNET]]["cpus"].pop()
                            for _ in range(res_config["num_cpus"]*num_local_res_replicas)],
            allocated_gpus=[addr_resource_map[name_addr_map[TF_RESNET]]["gpus"].pop()
                            for _ in range(num_local_res_replicas)]
        )
        addr_config_map[name_addr_map[TF_RESNET]].append(local_res_node_conf)
        if num_remote_res_replicas > 0:
            print("adding a remote replica")
            remote_res_node_conf = get_heavy_node_config(
                model_name=TF_RESNET,
                batch_size=int(res_config["batch_size"]),
                num_replicas=num_remote_res_replicas,
                cpus_per_replica=res_config["num_cpus"],
                allocated_cpus=[addr_resource_map[inception_addr]["cpus"].pop()
                                for _ in range(res_config["num_cpus"]*num_remote_res_replicas)],
                allocated_gpus=[addr_resource_map[inception_addr]["gpus"].pop()
                                for _ in range(num_remote_res_replicas)],
                remote_addr=name_addr_map[TF_RESNET])
            addr_config_map[name_addr_map[TF_RESNET]].append(remote_res_node_conf)
        # Provision KSVM
        addr_config_map[name_addr_map[TF_RESNET]].append(get_heavy_node_config(
                    model_name=TF_KERNEL_SVM,
                    batch_size=int(node_configs[TF_KERNEL_SVM]["batch_size"]),
                    num_replicas=node_configs[TF_KERNEL_SVM]["num_replicas"],
                    cpus_per_replica=node_configs[TF_KERNEL_SVM]["num_cpus"],
                    allocated_cpus=[addr_resource_map[name_addr_map[TF_RESNET]]["cpus"].pop()
                                    for _ in range(node_configs[TF_KERNEL_SVM]["num_cpus"]*node_configs[TF_KERNEL_SVM]["num_replicas"])],
                    allocated_gpus=None
                ))



        # Now check if we can fit resnet and inception on same machine
        incept_config = node_configs[INCEPTION_FEATS]
        if incept_config["num_replicas"] + num_local_res_replicas < MAX_GPUS_PER_NODE:
            assert num_remote_res_replicas == 0
            name_addr_map[INCEPTION_FEATS] = name_addr_map[TF_RESNET]


        addr_config_map[name_addr_map[INCEPTION_FEATS]].append(get_heavy_node_config(
                    model_name=INCEPTION_FEATS,
                    batch_size=int(node_configs[INCEPTION_FEATS]["batch_size"]),
                    num_replicas=node_configs[INCEPTION_FEATS]["num_replicas"],
                    cpus_per_replica=node_configs[INCEPTION_FEATS]["num_cpus"],
                    allocated_cpus=[addr_resource_map[name_addr_map[INCEPTION_FEATS]]["cpus"].pop()
                                    for _ in range(node_configs[INCEPTION_FEATS]["num_cpus"]*node_configs[INCEPTION_FEATS]["num_replicas"])],
                    allocated_gpus=[addr_resource_map[name_addr_map[INCEPTION_FEATS]]["gpus"].pop()
                                    for _ in range(node_configs[INCEPTION_FEATS]["num_replicas"])],
                )
        )
        addr_config_map[name_addr_map[INCEPTION_FEATS]].append(get_heavy_node_config(
                    model_name=TF_LOG_REG,
                    batch_size=int(node_configs[TF_LOG_REG]["batch_size"]),
                    num_replicas=node_configs[TF_LOG_REG]["num_replicas"],
                    cpus_per_replica=node_configs[TF_LOG_REG]["num_cpus"],
                    allocated_cpus=[addr_resource_map[name_addr_map[INCEPTION_FEATS]]["cpus"].pop()
                                    for _ in range(node_configs[TF_LOG_REG]["num_cpus"]*node_configs[TF_LOG_REG]["num_replicas"])],
                    allocated_gpus=None
                )
        )
    except IndexError:
        msg = "Ran out of available GPUs"
        logger.exception(msg)
        raise BenchmarkConfigurationException(msg)

    # Delete any addresses with no nodes assigned them, so we don't start the ZMQ frontend on them
    # unnecessarily
    for a in addr_config_map.keys():
        if len(addr_config_map[a]) == 0:
            del addr_config_map[a]
    # Sanity check
    for _, a in name_addr_map.iteritems():
        assert a in addr_config_map



    return name_addr_map, addr_config_map


def run_experiment_for_config(config, orig_config, resnet_addr, inception_addr, cv):
    try:
        name_addr_map, addr_config_map = assign_models_to_nodes(resnet_addr, inception_addr, config)
    except BenchmarkConfigurationException as e:
        logger.error("Error provisioning for requested configuration. Skipping.\n"
                     "Reason: {reason}\nBad config was:\n{conf}".format(reason=e, conf=config))
        return None
    print("\n\n\nADDR CONFIG MAP:\n{}".format(json.dumps(dict([(a, [c.__dict__ for c in cs]) for a, cs in addr_config_map.iteritems()]), indent=2)))
    print("NAME ADDR MAP:\n:{}".format(json.dumps(name_addr_map, indent=2)))

    lam = config["lam"]
    config["cv"] = cv
    slo = config["slo"]
    cost = config["cost"]
    utilization = config["utilization"]
    config["deltas_file_path"] = get_arrival_proc_file(lam, cv)
    config["deltas_file_md5sum"] = hash_file(config["deltas_file_path"])

    print("RUNNING SLO: {}, CV: {}, LAMBDA: {}".format(slo, cv, lam))

    if "latency_percentage" not in config:
        config["latency_percentage"] = 1.0
    # latency_perc = config["latency_percentage"]

    results_dir = "no_netcalc_e2e_sys_comp/pipeline_one/util_{}".format(utilization)
    reps_str = "_".join(["{name}-{reps}".format(name=c["name"], reps=c["num_replicas"])
                         for c in config["node_configs"].values()])
    results_fname = "aws_cv_{cv}_slo_{slo}_lambda_{lam}_cost_{cost}_reps_{reps_str}".format(
        lam=lam, cv=cv, slo=slo, cost=cost, reps_str=reps_str)


    # For client on standalone machine
    client_cpu_strs = [
        "0,1,2,3,4,5,6,7,32,33,34,35,36,37,38,39",
        "16,17,18,19,20,21,22,23,48,49,50,51,52,43,54,55"
    ]

    num_clients = 2
    trial_length = 2000

    throughput_results, models_to_replicate = run_e2e(
        addr_config_map, name_addr_map, trial_length, "../../release/src/inferline_client/image_driver_one",
        client_cpu_strs, lam, cv, num_clients, slo)
    if models_to_replicate is None:
        driver_utils.save_results_cpp_client(
            dict([(a, [c.__dict__ for c in cs]) for a, cs in addr_config_map.iteritems()]),
            throughput_results,
            None,
            results_dir,
            prefix=results_fname,
            orig_config=orig_config,
            used_config=config)
    # else:
    #     new_config = config.copy()
    #     for m in models_to_replicate:
    #         new_config["node_configs"][m]["num_replicas"] += 1
    #     logger.info(("GREPTHISAAAAA: Rerunning with more replicas of: {m}"
    #                  "OLD CONFIG:\n{old}\nNEW CONFIG: {new}").format(
    #                      m=models_to_replicate,
    #                      old=config,
    #                      new=new_config))
    #     run_experiment_for_config(new_config, orig_config)


if __name__ == "__main__":

    base_path = os.path.expanduser("/home/ubuntu/plots-model-comp-paper/experiments/e2e_sys_comp_no_netcalc/pipeline_one/util_1.0")

    config_paths = [
        "aws_image_driver_one_ifl_configs_slo_0.5_util_1.0.json",
        # "aws_image_driver_one_ifl_configs_slo_1.0_util_1.0.json"
    ]

    config_paths = [os.path.join(base_path, c) for c in config_paths]
    for config_path in config_paths:
        print(config_path)
        with open(os.path.abspath(os.path.expanduser(config_path)), "r") as f:
            provided_configs = json.load(f)

        for config in provided_configs:
            # for cv in [0.1, 0.0]:
            for cv in [4.0, 1.0]:
                run_experiment_for_config(
                    config, config.copy(),
                    os.environ["RESNET_CLIPPER_ADDR"],
                    os.environ["INCEPTION_CLIPPER_ADDR"], cv)
    sys.exit(0)
