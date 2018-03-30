import argparse
import subprocess32 as subprocess
import os
import sys
# import numpy as np
import time
import logging
import json
from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
from containerized_utils import driver_utils

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__)) 

DEFAULT_OUTPUT = "TIMEOUT"
CLIPPER_ADDRESS = "localhost"

LANG_DETECT_MODEL_APP_NAME = "tf-lang-detect"
NMT_MODEL_APP_NAME = "tf-nmt"
LSTM_MODEL_APP_NAME = "tf-lstm"

def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus,
                          input_size=None):

    if model_name == LSTM_MODEL_APP_NAME:
        image = "gcr.io/clipper-model-comp/tf-lstm:bench"
        return driver_utils.HeavyNodeConfig(name=LSTM_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

    elif model_name == NMT_MODEL_APP_NAME:
        image = "gcr.io/clipper-model-comp/tf-nmt:bench"
        return driver_utils.HeavyNodeConfig(name=NMT_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

    elif model_name == LANG_DETECT_MODEL_APP_NAME:
        image = "gcr.io/clipper-model-comp/tf-lang-detect:bench"
        return driver_utils.HeavyNodeConfig(name=LANG_DETECT_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

def setup_clipper(configs):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    cl.stop_all()
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
    return CLIPPER_ADDRESS


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


def get_profiler_stats(metrics_json):
    mean_latencies = {}
    p99_latencies = {}
    thrus = {}
    counts = {}

    hists = metrics_json["histograms"]
    for h in hists:
        if "prediction_latency" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = float(h[name]["mean"]) / 1000.0
            p99 = float(h[name]["p99"]) / 1000.0
            mean_latencies[model] = round(float(mean), 3)
            p99_latencies[model] = round(float(p99), 3)
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
    return (mean_latencies, p99_latencies, thrus, counts)


def print_stats(client_metrics, clipper_metrics):
    results_dict = {}
    results_dict["batch_sizes"] = get_clipper_batch_sizes(clipper_metrics)
    results_dict["queue_sizes"] = get_clipper_queue_sizes(clipper_metrics)
    results_dict["clipper_thrus"] = get_clipper_thruputs(clipper_metrics)
    results_dict["clipper_counts"] = get_clipper_counts(clipper_metrics)
    results_dict["client_mean_lats"], results_dict["client_p99_lats"], \
        results_dict["client_thrus"], results_dict["client_counts"] = \
        get_profiler_stats(client_metrics)
    # logger.info(("\nClient thrus: {client_thrus}, clipper thrus: {clipper_thrus} "
    #              "\nclient counts: {client_counts}, clipper counts: {clipper_counts}, "
    #              "\nclient p99 lats: {client_p99_lats}, client mean lats: {client_mean_lats} "
    #              "\nqueue sizes: {queue_sizes}, "
    #              "batch sizes: {batch_sizes}\n").format(**results_dict))
    logger.info(("\nThroughput: {client_thrus}\nP99 lat: {client_p99_lats}"
                 "\nMean lat: {client_mean_lats} "
                 "\nBatches: {batch_sizes}\nQueues: {queue_sizes}\n").format(**results_dict))
    return results_dict


def load_metrics(client_path, clipper_path):
    with open(client_path, "r") as client_file, \
            open(clipper_path, "r") as clipper_file:
        client_metrics_str = client_file.read().strip()
        clipper_metrics_str = clipper_file.read().strip()
        if client_metrics_str[-1] == ",":
            client_metrics_str = client_metrics_str.rstrip(",")
            client_metrics_str += "]"
        if clipper_metrics_str[-1] == ",":
            clipper_metrics_str = clipper_metrics_str.rstrip(",")
            clipper_metrics_str += "]"
        try:
            client_metrics = json.loads(client_metrics_str)
        except ValueError as e:
            # logger.warn("Unable to parse client metrics: {}. Skipping...".format(e))
            return None
        try:
            clipper_metrics = json.loads(clipper_metrics_str)
        except ValueError as e:
            # logger.warn("Unable to parse clipper metrics: {}. Skipping...".format(e))
            return None
    return client_metrics, clipper_metrics


def load_lineage(lineage_path):
    with open(lineage_path, "r") as f:
        parsed = [json.loads(l) for l in f]
    return parsed


def run_profiler(configs, trial_length, driver_path, profiler_cores_str, target_throughput):
    clipper_address = setup_clipper(configs)
    clipper_address = CLIPPER_ADDRESS
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    time.sleep(30)
    log_dir = "/tmp/text_driver_one_profiler_logs_{ts:%y%m%d_%H%M%S}".format(ts=datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def run(target_throughput, num_trials, name, arrival_process):
        cl.drain_queues()
        time.sleep(10)
        cl.drain_queues()
        time.sleep(10)
        arrival_delay_file = os.path.abspath("arrival_deltas_ms.timestamp")
        log_path = os.path.join(log_dir, "{n}-{t}-{p}".format(n=name,
                                                              t=target_throughput,
                                                              p=arrival_process))
        workload_path = os.path.join(CURR_DIR, "lang_detect_workload", "workload.txt");
        cmd = ["numactl", "-C", profiler_cores_str,
               os.path.abspath(driver_path),
               "--target_throughput={}".format(target_throughput),
               "--request_distribution={}".format(arrival_process),
               "--trial_length={}".format(trial_length),
               "--num_trials={}".format(num_trials),
               "--log_file={}".format(log_path),
               "--clipper_address={}".format(clipper_address),
               "--request_delay_file={}".format(arrival_delay_file),
               "--workload_path={}".format(workload_path)]

        logger.info("Driver command: {}".format(" ".join(cmd)))
        client_path = "{p}-client_metrics.json".format(p=log_path)
        clipper_path = "{p}-clipper_metrics.json".format(p=log_path)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            recorded_trials = 0
            summary_results = []
            while recorded_trials < num_trials:
                time.sleep(5)
                if proc.poll() is not None:
                    if proc.returncode != 0:
                        logger.warn("Driver process finished with return code {}".format(
                            proc.returncode))
                        break
                    # else:
                    #     logger.warn("Driver process finished without enough trials. "
                    #                 "Expected {num}, recorded {rec}".format(num=num_trials,
                    #                                                         rec=recorded_trials))
                if not (os.path.exists(client_path) and os.path.exists(clipper_path)):
                    logger.info("metrics don't exist yet")
                    continue
                loaded_metrics = load_metrics(client_path, clipper_path)
                if loaded_metrics is not None:
                    client_metrics, clipper_metrics = loaded_metrics
                else:
                    continue
                client_trials = len(client_metrics)
                clipper_trials = len(clipper_metrics)
                if clipper_trials != client_trials:
                    logger.warn(("Clipper trials ({}) and client trials ({}) not the "
                                 "same length").format(clipper_trials, client_trials))
                else:
                    new_recorded_trials = clipper_trials
                    if new_recorded_trials > recorded_trials:
                        recorded_trials = new_recorded_trials
                        summary_results.append(print_stats(client_metrics[-1],
                                                           clipper_metrics[-1]))

            # prof_stdout = proc.stdout.read().strip()
            # if len(prof_stdout) > 0:
            #     logger.info("Profiler stdout: {}".format(prof_stdout))
            prof_stderr = proc.stderr.read().strip()
            if len(prof_stderr) > 0:
                logger.info("stderr: {}".format(prof_stderr))
            try:
                loaded_metrics = load_metrics(client_path, clipper_path)
                # lineage = load_lineage(lineage_path)
                if loaded_metrics is not None:
                    client_metrics, clipper_metrics = loaded_metrics
                    return driver_utils.Results(client_metrics,
                                                clipper_metrics,
                                                summary_results,
                                                None)
                else:
                    logger.error("Error loading final metrics")
            except ValueError as e:
                logger.error("Unable to parse final metrics")
                raise e

    run(target_throughput, 5, "warmup", "constant")
    throughput_results = run(target_throughput, 20, "throughput", "poisson")
    cl.stop_all()
    return throughput_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper text driver 1')
    parser.add_argument('-g', '--num_gpus', type=int, default=4, help="The number of GPUs available for use")

    args = parser.parse_args()

    lstm_batch_size = 16
    nmt_batch_size = 2
    lang_detect_batch_size = 16

    model_cpus = range(4, 11)
    model_gpus = range(args.num_gpus)

    def get_cpus(num):
        return [model_cpus.pop() for _ in range(num)]
    def get_gpus(num):
        return [model_gpus.pop() for _ in range(num)]

    lstm_replicas = 1
    nmt_replicas = 1
    lang_detect_replicas = 1

    configs = [
        get_heavy_node_config(
                model_name=LSTM_MODEL_APP_NAME,
                batch_size=lstm_batch_size,
                num_replicas=lstm_replicas,
                cpus_per_replica=1,
                allocated_cpus=get_cpus(lstm_replicas),
                allocated_gpus=get_gpus(lstm_replicas),
            ),
        get_heavy_node_config(
                model_name=NMT_MODEL_APP_NAME,
                batch_size=nmt_batch_size,
                num_replicas=nmt_replicas,
                cpus_per_replica=1,
                allocated_cpus=get_cpus(nmt_replicas),
                allocated_gpus=get_gpus(nmt_replicas),
            ),
        get_heavy_node_config(
                model_name=LANG_DETECT_MODEL_APP_NAME,
                batch_size=lang_detect_batch_size,
                num_replicas=lang_detect_replicas,
                cpus_per_replica=1,
                allocated_cpus=get_cpus(lang_detect_replicas),
                allocated_gpus=None,
            )
    ]

    target_throughput = 1000

    throughput_results = run_profiler(
        configs, 2000, "../../release/src/inferline_client/text_driver_one", "11,27,12,28", target_throughput)
    fname = "cpp-aws-p2-{ls}-lstm-{nm}-nmt-{ld}-lang_detect".format(
        ls=lstm_replicas,
        nmt=nmt_replicas,
        lang_detect=lang_detect_replicas)
    results_dir = "text_driver_one_e2e"
    driver_utils.save_results_cpp_client(
        configs,
        throughput_results,
        None,
        results_dir,
        prefix=fname)
    sys.exit(0)
