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
import argparse

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

# REMOTE_ADDR = None
ALL_REMOTE_ADDRS = None

RESNET_CLIPPER_ADDR = "172.30.3.111"
INCEPTION_CLIPPER_ADDR = RESNET_CLIPPER_ADDR
# INCEPTION_CLIPPER_ADDR = "172.30.3.156"


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
        for c in configs:
            driver_utils.setup_heavy_node(cl, c, DEFAULT_OUTPUT)
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
    for metrics_json in cur_client_metrics:
        latencies = {}
        thrus = {}
        counts = {}

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
        all_counts.append(counts)
    agg_p99_latency = {}
    agg_mean_latency = {}
    for model in all_latencies[0]:
        lats = np.hstack([c[model] for c in all_latencies])
        agg_p99_latency[model] = round(np.percentile(lats, 99), 3)
        agg_mean_latency[model] = round(np.mean(lats), 3)

    agg_thrus = {}
    for model in all_thrus[0]:
        total_thru = np.sum([c[model] for c in all_thrus])
        agg_thrus[model] = total_thru

    agg_counts = {}
    for model in all_counts[0]:
        total_count = np.sum([c[model] for c in all_counts])
        agg_counts[model] = total_count

    return (agg_mean_latency, agg_p99_latency, agg_thrus, agg_counts)


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
        results_dict["client_thrus"], results_dict["client_counts"] = \
        get_profiler_stats(cur_client_metrics)
    logger.info(("\nThroughput: {client_thrus}\nP99 lat: {client_p99_lats}"
                "\nMean lat: {client_mean_lats}").format(**results_dict))
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

def run_e2e(addr_config_map, trial_length, driver_path, profiler_cores_strs, lam, cv, num_clients):
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

    def run(target_throughput, num_trials, name, arrival_process):
        for cl in cls:
            cl.drain_queues()
        time.sleep(10)
        for cl in cls:
            cl.drain_queues()
        time.sleep(10)
        if cv == 1:
            arrival_file_name = "{lam}.deltas".format(lam=lam)
        else:
            arrival_file_name = "{lam}_{cv}.deltas".format(lam=lam, cv=cv)
        arrival_delay_file = os.path.join(("/home/ubuntu/plots-model-comp-paper/experiments/"
                                          "cached_arrival_processes/{f}").format(f=arrival_file_name))
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
                    "--clipper_address_resnet={}".format(RESNET_CLIPPER_ADDR),
                    "--clipper_address_inception={}".format(INCEPTION_CLIPPER_ADDR),
                    "--request_delay_file={}".format(arrival_delay_file)]
                if client_num == 0:
                    cmd.append("--get_clipper_metrics")

                logger.info("Driver command: {}".format(" ".join(cmd)))
                client_path = "{p}-client_metrics.json".format(p=log_path)
                lineage_paths = {m: "{p}-{m}-query_lineage.txt".format(m=m, p=log_path)
                                for m in [TF_RESNET, INCEPTION_FEATS, TF_KERNEL_SVM, TF_LOG_REG]}
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                procs[client_num] = (proc, log_path, client_path, lineage_paths)
            clipper_paths = ["{p}-clipper_metrics_resnet.json".format(p=log_path)]
            if len(addr_config_map) == 2:
                clipper_paths.append("{p}-clipper_metrics_incept.json".format(p=log_path))

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
                        summary_results.append(print_stats(cur_client_metrics, new_recorded_trials - 1))
                        print_stats_this_iter = True

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
                    # lineages = {name: load_lineage(p) for name, p in lineage_paths.items()}
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
                                        None)
        except Exception as e:
            logger.exception(e)
        finally:
            for c in procs:
                procs[c][0].terminate()

    run(100, 3, "warmup", "constant")
    throughput_results = run(0, 25, "throughput", "file")

    for cl in cls:
        cl.stop_all(remote_addrs=ALL_REMOTE_ADDRS)
    return throughput_results


class BenchmarkConfigurationException(Exception):
    pass


def run_experiment_for_config(config):
    # res_cpus = range(4, 16)
    res_cpus = range(8, 16)
    res_gpus = range(4)

    incept_cpus = range(4, 16)
    incept_gpus = range(4)

    if RESNET_CLIPPER_ADDR == INCEPTION_CLIPPER_ADDR:
        incept_cpus = res_cpus
        incept_gpus = res_gpus

    incept_branch = [INCEPTION_FEATS, TF_LOG_REG]
    res_branch = [TF_RESNET, TF_KERNEL_SVM]

    def get_cpus(num, model):
        try:
            if model in incept_branch:
                return [incept_cpus.pop() for _ in range(num)]
            elif model in res_branch:
                return [res_cpus.pop() for _ in range(num)]
        except IndexError:
            msg = "Ran out of out available CPUs"
            logger.error(msg)
            raise BenchmarkConfigurationException(msg)

    def get_gpus(num, gpu_type, model):
        if gpu_type == "none":
            return None
        else:
            assert gpu_type == "v100"
            try:
                if model in incept_branch:
                    return [incept_gpus.pop() for _ in range(num)]
                elif model in res_branch:
                    return [res_gpus.pop() for _ in range(num)]
            except IndexError:
                msg = "Ran out of available GPUs"
                logger.error(msg)
                raise BenchmarkConfigurationException(msg)

    try:
        node_configs = []
        addr_config_map = {RESNET_CLIPPER_ADDR: [], INCEPTION_CLIPPER_ADDR: []}
        for name, c in config["node_configs"].iteritems():
            if name in [TF_RESNET, TF_KERNEL_SVM]:
                node = get_heavy_node_config(model_name=c["name"],
                                    batch_size=int(c["batch_size"]),
                                    num_replicas=c["num_replicas"],
                                    cpus_per_replica=c["num_cpus"],
                                    allocated_cpus=get_cpus(c["num_cpus"]*c["num_replicas"], name),
                                    allocated_gpus=get_gpus(c["num_replicas"], c["gpu_type"], name))
                node_configs.append(node)
                addr_config_map[RESNET_CLIPPER_ADDR].append(node)
            if name in [INCEPTION_FEATS, TF_LOG_REG]:
                node = get_heavy_node_config(model_name=c["name"],
                                    batch_size=int(c["batch_size"]),
                                    num_replicas=c["num_replicas"],
                                    cpus_per_replica=c["num_cpus"],
                                    allocated_cpus=get_cpus(c["num_cpus"]*c["num_replicas"], name),
                                    allocated_gpus=get_gpus(c["num_replicas"], c["gpu_type"], name))
                node_configs.append(node)
                addr_config_map[INCEPTION_CLIPPER_ADDR].append(node)


        # node_configs = [
        #                 for c in config["node_configs"].values()]
        # addr_config_map = {RESNET_CLIPPER_ADDR: [], INCEPTION_CLIPPER_ADDR: []}
        # for n in node_configs:
        #     if n.name in [TF_RESNET, TF_KERNEL_SVM]:
        #         addr_config_map[RESNET_CLIPPER_ADDR].append(n)
        #     elif n.name in [INCEPTION_FEATS, TF_LOG_REG]:
        #         addr_config_map[INCEPTION_CLIPPER_ADDR].append(n)
        # print(addr_config_map)


    except BenchmarkConfigurationException as e:
        logger.error("Error provisioning for requested configuration. Skipping.\n"
                     "Reason: {reason}\nBad config was:\n{conf}".format(reason=e, conf=config))
        return None
    lam = config["lam"]
    cv = config["cv"]
    slo = config["slo"]
    cost = config["cost"]

    results_dir = "image_driver_one_slo_{slo}_cv_{cv}-DEBUG".format(slo=slo, cv=cv)
    reps_str = "_".join(["{name}-{reps}".format(name=c["name"], reps=c["num_replicas"])
                         for c in config["node_configs"].values()])
    results_fname = "aws_lambda_{lam}_cost_{cost}_{reps_str}".format(
        lam=lam, cost=cost, reps_str=reps_str)

    # if RESNET_CLIPPER_ADDR == INCEPTION_CLIPPER_ADDR:
    #     client_cpu_str = "4,20,5,21"
    # else:
    #     client_cpu_str = "4,5,6,7,8,9,20,21,22,23,24,25"

    # For client on standalone machine
    client_cpu_strs = [
        "0,1,2,3,4,5,6,7,32,33,34,35,36,37,38,39",
        "16,17,18,19,20,21,22,23,48,49,50,51,52,43,54,55"
    ]

    # num_clients = 2
    num_clients = 1

    throughput_results = run_e2e(
        addr_config_map, 2000, "../../release/src/inferline_client/image_driver_one",
        client_cpu_strs, int(lam / num_clients), cv, num_clients)
    driver_utils.save_results_cpp_client(
        node_configs,
        throughput_results,
        None,
        results_dir,
        prefix=results_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Image Driver 1 experiments')
    parser.add_argument('-c', '--config_path', type=str, help="Path to config JSON file generated by optimizer")
    args = parser.parse_args()

    with open(os.path.abspath(os.path.expanduser(args.config_path)), "r") as f:
        provided_configs = json.load(f)

    for config in provided_configs:
        run_experiment_for_config(config)
        break
    sys.exit(0)
