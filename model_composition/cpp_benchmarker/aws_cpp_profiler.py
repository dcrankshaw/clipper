import subprocess32 as subprocess
import os
# import sys
import numpy as np
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

DEFAULT_OUTPUT = "TIMEOUT"
CLIPPER_ADDRESS = "localhost"

RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"


def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus):

    if model_name == "alexnet":
        image = "gcr.io/clipper-model-comp/pytorch-alexnet:bench"
        return driver_utils.HeavyNodeConfig(name="alexnet",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )

    elif model_name == "res50":
        image = "gcr.io/clipper-model-comp/pytorch-res50:bench"
        return driver_utils.HeavyNodeConfig(name="res50",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )

    elif model_name == "res152":
        image = "gcr.io/clipper-model-comp/pytorch-res152:bench"
        return driver_utils.HeavyNodeConfig(name="res152",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )


def setup_clipper(configs):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
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
            mean = h[name]["mean"]
            mean_batch_sizes[model] = round(float(mean), 2)
    return mean_batch_sizes


def get_clipper_queue_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_queue_sizes = {}
    for h in hists:
        if "queue_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = h[name]["mean"]
            mean_queue_sizes[model] = round(float(mean), 2)
    return mean_queue_sizes


def get_clipper_thruputs(metrics_json):
    meters = metrics_json["meters"]
    thrus = {}
    for m in meters:
        if "prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            rate = m[name]["rate"]
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
            mean = h[name]["mean"]
            p99 = h[name]["p99"]
            mean_latencies[model] = round(float(mean), 3)
            p99_latencies[model] = round(float(p99), 3)
    meters = metrics_json["meters"]
    for m in meters:
        if "prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            model = name.split(":")[0]
            rate = m[model]["rate"]
            thrus[model] = round(float(rate), 5)
    counters = metrics_json["counters"]
    counts = {}
    for c in counters:
        if "num_predictions" not in c.keys()[0]:
            name = c.keys()[0]
            model = name.split(":")[0]
            count = c[model]["count"]
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
    logger.info(("Client thrus: {client_thrus}, clipper thrus: {clipper_thrus "
                 "client counts: {client_counts}, clipper counts: {clipper_counts}, "
                 "client p99 lats: {client_p99_lats}, client mean lats: {client_mean_lats} "
                 "queue sizes: {queue_sizes}, batch sizes: {batch_sizes}").format(**results_dict))
    return results_dict


def run_profiler(config, trial_length, driver_path, input_size):
    clipper_address = setup_clipper([config, ])
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    time.sleep(30)
    log_dir = "{name}_profiler_logs_{ts:%y%m%d_%H%M%S}".format(name=config.name, ts=datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def run(delay_micros, num_trials, name):
        cl.drain_queues()
        time.sleep(10)
        cl.drain_queues()
        time.sleep(10)
        log_path = os.path.join(log_dir, "name-{n}-delay-{d}".format(n=name, d=delay_micros))
        # TODO: pin the driver to certain cores
        cmd = [os.path.abspath(driver_path), "--name={}".format(config.name),
               "--input_type={}".format(config.input_type),
               "--input_size={}".format(input_size),
               "--request_delay_micros={}".format(delay_micros),
               "--trial_length={}".format(trial_length),
               "--num_trials={}".format(num_trials),
               "--log_file={}".format(log_path),
               "--clipper_address={}".format(clipper_address)]
        logger.info("Driver command: {}".format(" ".join(cmd)))
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            recorded_trials = 0
            all_results = []
            while recorded_trials < num_trials:
                time.sleep(15)
                if proc.poll() is not None:
                    logger.info("Driver process finished with return code {}".format(
                        proc.returncode))
                    break
                client_path = "{p}-client_metrics.json".format(p=log_path)
                clipper_path = "{p}-clipper_metrics.json".format(p=log_path)
                if not (os.path.exists(client_path) and os.path.exists(clipper_path)):
                    logger.info("metrics don't exist yet")
                    continue
                with open(client_path, "r") as client_file, \
                        open(clipper_path, "r") as clipper_file:
                    client_metrics_str = client_file.read()
                    clipper_metrics_str = clipper_file.read()
                    if client_metrics_str[-1] != "]":
                        client_metrics_str += "]"
                    if clipper_metrics_str[-1] != "]":
                        clipper_metrics_str += "]"
                    try:
                        client_metrics = json.loads(client_metrics_str)
                        clipper_metrics = json.loads(client_metrics_str)
                        client_trials = len(client_metrics)
                        clipper_trials = len(clipper_metrics)
                        if clipper_trials != client_trials:
                            logger.warn(("Clipper trials ({}) and client trials ({}) not the "
                                         "same length").format(clipper_trials, client_trials))
                        else:
                            new_recorded_trials = clipper_trials
                            if new_recorded_trials > recorded_trials:
                                recorded_trials = new_recorded_trials
                                all_results.append(print_stats(client_metrics[-1],
                                                               clipper_metrics[-1]))
                    except ValueError as e:
                        logger.warn("Unable to parse metrics. Skipping for now. {}".format(e))

            logger.info("stdout: {}".format(proc.stdout.read()))
            logger.info("stderr: {}".format(proc.stderr.read()))
            return all_results

    run(1000, 5, "warmup")
    # init_results = run(1000, 10, "init")
    # mean_thruput = np.mean([r["client_thrus"] for r in init_results][1:])
    # steady_state_delay = round(1.0 / mean_thruput * 1000.0 * 1000.0)
    # logger.info("Setting delay to {}".format(steady_state_delay))
    # run(steady_state_delay, 20, "steady_state")


if __name__ == "__main__":

    config = get_heavy_node_config(
        model_name=RES50,
        batch_size=8,
        num_replicas=1,
        cpus_per_replica=1,
        allocated_cpus=range(4, 8),
        allocated_gpus=range(4)
    )

    run_profiler(config, 500, "../../release/src/inferline_client/profiler", 299*299*3)
