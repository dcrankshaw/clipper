from __future__ import print_function
import json
import logging
import os
import datetime
import requests
from scipy.stats import linregress
import numpy as np

logger = logging.getLogger(__name__)

INCREASING = "increasing"
CONVERGED_HIGH = "converged_high"
CONVERGED_LOW = "converged_low"
DECREASING = "decreasing"
CONVERGED = "converged"
UNKNOWN = "unknown"


class Results(object):
    def __init__(self, client_metrics, clipper_metrics, summary_metrics, lineage):
        self.client_metrics = client_metrics
        self.clipper_metrics = clipper_metrics
        self.summary_metrics = summary_metrics
        self.lineage = lineage

    def get_dict(self):
        return self.__dict__


class HeavyNodeConfig(object):
    def __init__(self,
                 name,
                 input_type,
                 model_image,
                 allocated_cpus,
                 cpus_per_replica,
                 num_replicas,
                 gpus,
                 batch_size,
                 use_nvidia_docker,
                 slo=5000000,
                 input_size=-1,
                 no_diverge=False):
        self.name = name
        self.cloud = "aws"
        self.input_type = input_type
        self.model_image = model_image
        self.allocated_cpus = allocated_cpus
        self.cpus_per_replica = cpus_per_replica
        self.slo = slo
        self.num_replicas = num_replicas
        self.gpus = gpus
        self.batch_size = batch_size
        self.use_nvidia_docker = use_nvidia_docker
        self.input_size = input_size
        self.instance_type = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type").text
        if len(gpus) == 0:
            self.gpus_per_replica = 0
        else:
            self.gpus_per_replica = 1
        self.no_diverge = no_diverge

    def to_json(self):
        return json.dumps(self.__dict__)


def setup_heavy_node(clipper_conn, config, default_output="TIMEOUT"):
    clipper_conn.register_application(name=config.name,
                                      default_output=default_output,
                                      slo_micros=config.slo,
                                      input_type=config.input_type)

    clipper_conn.deploy_model(name=config.name,
                              version=1,
                              image=config.model_image,
                              input_type=config.input_type,
                              num_replicas=config.num_replicas,
                              batch_size=config.batch_size,
                              gpus=config.gpus,
                              allocated_cpus=config.allocated_cpus,
                              cpus_per_replica=config.cpus_per_replica,
                              use_nvidia_docker=config.use_nvidia_docker)

    clipper_conn.link_model_to_app(app_name=config.name, model_name=config.name)


class HeavyNodeConfigGCP(object):
    def __init__(self,
                 name,
                 input_type,
                 model_image,
                 cpus_per_replica,
                 num_replicas,
                 gpu_type,
                 batch_size,
                 slo=5000000,
                 input_size=-1,
                 no_diverge=False):
        self.name = name
        self.cloud = "gcp"
        self.input_type = input_type
        self.model_image = model_image
        self.cpus_per_replica = cpus_per_replica
        self.slo = slo
        self.num_replicas = num_replicas
        self.gpu_type = gpu_type
        self.batch_size = batch_size
        self.input_size = input_size
        self.gpu_type = gpu_type
        self.no_diverge = no_diverge

    def to_json(self):
        return json.dumps(self.__dict__)


def setup_heavy_node_gcp(clipper_conn, config, default_output="TIMEOUT"):
    clipper_conn.register_application(name=config.name,
                                      default_output=default_output,
                                      slo_micros=config.slo,
                                      input_type=config.input_type)

    clipper_conn.deploy_model(name=config.name,
                              version=1,
                              image=config.model_image,
                              input_type=config.input_type,
                              num_replicas=config.num_replicas,
                              batch_size=config.batch_size,
                              gpu_type=config.gpu_type,
                              num_cpus=config.cpus_per_replica)

    clipper_conn.link_model_to_app(app_name=config.name, model_name=config.name)


def save_results(configs, client_metrics, init_metrics, results_dir,
                 prefix="results",
                 container_metrics=None):
    """
    Parameters
    ----------
    configs : list(HeavyNodeConfig)
        The configs for any models deployed


    """

    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created experiments directory: %s" % results_dir)

    if "all_lats" not in client_metrics[0]:
        raise Exception("No latencies list found under key \"all_lats\"."
                        " Please update your driver to include all latencies so we can"
                        " plot the latency CDF")
    else:
        for c in client_metrics:
            all_lats_strs = [json.dumps(list(l)) for l in c["all_lats"]]
            c["all_lats"] = all_lats_strs

    results_obj = {
        "node_configs": [c.__dict__ for c in configs],
        "client_metrics": client_metrics,
        "init_client_metrics": init_metrics,
    }
    if container_metrics is not None:
        results_obj["container_metrics"] = container_metrics
    results_file = os.path.join(results_dir, "{prefix}-{ts:%y%m%d_%H%M%S}.json".format(
        prefix=prefix, ts=datetime.datetime.now()))
    with open(results_file, "w") as f:
        json.dump(results_obj, f, indent=4)
        logger.info("Saved results to {}".format(results_file))


def save_results_cpp_client(configs,
                            throughput_results,
                            latency_results,
                            results_dir,
                            prefix="results",
                            container_metrics=None):
    """
    Parameters
    ----------
    configs : list(HeavyNodeConfig)
        The configs for any models deployed


    """

    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created experiments directory: %s" % results_dir)

    # if "all_lats" not in client_metrics[0]:
    #     raise Exception("No latencies list found under key \"all_lats\"."
    #                     " Please update your driver to include all latencies so we can"
    #                     " plot the latency CDF")
    # else:
    #     for c in client_metrics:
    #         all_lats_strs = [json.dumps(list(l)) for l in c["all_lats"]]
    #         c["all_lats"] = all_lats_strs

    results_obj = {
        "node_configs": [c.__dict__ for c in configs]
    }
    if throughput_results is not None:
        results_obj["throughput_results"] = throughput_results.get_dict()
    if latency_results is not None:
        results_obj["latency_results"] = latency_results.get_dict(),
    if container_metrics is not None:
        results_obj["container_metrics"] = container_metrics

    results_file = os.path.join(results_dir, "{prefix}-{ts:%y%m%d_%H%M%S}.json".format(
        prefix=prefix, ts=datetime.datetime.now()))
    with open(results_file, "w") as f:
        json.dump(results_obj, f, indent=4)
        logger.info("Saved results to {}".format(results_file))


def check_convergence_via_queue(stats, configs):
    """
    Returns
    -------
    boolean : whether we're in a steady state yet
    slope_sign : if we're not in a steady state, the sign of
        the slope of the function computing latency
        as a function of time. If the slope is negative, we're
        draining the queues and we should keep waiting. If the slope is
        positive, we're filling the queues faster than they can be drained
        and we should quit and increase the delay.

    """

    window_size = min(15, len(stats["mean_queue_sizes"]) - 4)
    # Raw is a list of dicts
    queue_sizes_raw = stats["mean_queue_sizes"][-1*window_size:]
    batch_sizes_raw = stats["mean_batch_sizes"][-1*window_size:]
    queue_sizes = {}
    batch_sizes = {}
    # Reshape to a dict of lists
    for model in queue_sizes_raw[0].keys():
        queue_sizes[model] = []
        batch_sizes[model] = []

    for entry in queue_sizes_raw:
        for model, size in entry.iteritems():
            queue_sizes[model].append(size)

    for entry in batch_sizes_raw:
        for model, size in entry.iteritems():
            batch_sizes[model].append(size)

    window_size = min(15, len(batch_sizes) - 4)

    desired_batch_sizes = {}
    for c in configs:
        desired_batch_sizes[c.name] = c.batch_size

    # Check that at least one model is using large batch sizes (this should be the bottleneck model)
    found_full_batches = False
    for model_name in desired_batch_sizes.keys():
        mean_of_means_batch_size = np.mean(batch_sizes[model_name])
        if mean_of_means_batch_size > desired_batch_sizes[model_name]*0.85:
            logger.info(("{name} is using full-size batches. Desired {desired}, "
                         "actual: {actual}").format(
                             name=model_name,
                             actual=mean_of_means_batch_size,
                             desired=float(desired_batch_sizes[model_name])))
            found_full_batches = True

    if not found_full_batches:
        logger.info("No nodes are using large batch sizes")
        return CONVERGED_LOW

    # Now check that none of the queues are overloaded
    for model_name, qs in queue_sizes.iteritems():
        mean_queue_size = np.mean(qs)
        std_queue_size = np.std(qs)
        # Check if queue behavior has stabilized
        if (mean_queue_size < 3.0*desired_batch_sizes[model_name] and
                std_queue_size < 1.5*desired_batch_sizes[model_name]):
            continue
        else:
            lr = linregress(x=range(len(qs)), y=qs)
            logger.info("{mname} regression results:\n{lr}".format(mname=model_name, lr=lr))
            # If pvalue less than 0.001, the line definitely has a slope
            if lr.pvalue < 0.01:
                if lr.slope > 0:
                    return INCREASING
                else:
                    return DECREASING
            else:
                # If the pvalues is greater than 0.1, the queue size is not changing but it's still
                # too high
                return CONVERGED_HIGH
    # If we've reached the end of the for loop and all the nodes queueing behavior is stable, return
    # converged
    return CONVERGED


def check_convergence(stats, configs, latency_upper_bound=None):
    """
    Returns
    -------
    boolean : whether we're in a steady state yet
    slope_sign : if we're not in a steady state, the sign of
        the slope of the function computing latency
        as a function of time. If the slope is negative, we're
        draining the queues and we should keep waiting. If the slope is
        positive, we're filling the queues faster than they can be drained
        and we should quit and increase the delay.

    """
    window_size = min(15, len(stats["p99_lats"]) - 4)
    p99_lats = stats["p99_lats"][-1*window_size:]

    mean_batch_sizes = {}
    for c in configs:
        mean_batch_sizes[c.name] = np.mean(
            [b[c.name] for b in stats["mean_batch_sizes"][-1*window_size:]])
    lr = linregress(x=range(len(p99_lats)), y=p99_lats)
    logger.info(lr)
    # pvalue checks against the null hypothesis that the
    # slope is 0. We are checking for the slope to be 0,
    # so we want to the null hypothesis to be true.

    # If pvalue less than 0.001, the line definitely has a slope
    if lr.pvalue < 0.001:
        if lr.slope > 0:
            return INCREASING
        else:
            return DECREASING
    elif lr.pvalue > 0.2:
        # Slope is 0, now check to see
        # if mean batch_sizes are less
        # than
        # configured batch size.

        if latency_upper_bound is not None:
            mean_p99_lats = np.mean(p99_lats)
            if mean_p99_lats > latency_upper_bound:
                logger.info(("Slope is 0 but p99 latency ({lat} s) is too high"
                             " for node {name}").format(lat=mean_p99_lats,
                                                        name=c.name))
                return CONVERGED_HIGH

        # If any of the nodes batch sizes are set to 1, skip the batch size check
        for c in configs:
            if c.batch_size == 1.0:
                return CONVERGED

        # We don't know which node is the bottleneck, so we check that all the nodes have batch
        # sizes slightly less than the configured batch size.
        for c in configs:
            if mean_batch_sizes[c.name] == c.batch_size:
                logger.info(("Slope is 0 but batch_sizes too big for "
                             "node {name}").format(name=c.name))
                return CONVERGED_HIGH
        return CONVERGED
    else:
        return UNKNOWN
