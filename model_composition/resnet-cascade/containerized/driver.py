# import sys
# import os
import argparse
import numpy as np
import time
# import base64
import logging
import json

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
# from io import BytesIO
# from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from multiprocessing import Process, Queue
from scipy.stats import linregress


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"


CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"


def setup_clipper(config):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="1-11")
    time.sleep(10)
    driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")
    return config


def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus):

    if model_name == "alexnet":
        return driver_utils.HeavyNodeConfig(name="alexnet",
                                            input_type="floats",
                                            model_image="model-comp/pytorch-alexnet",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )

    elif model_name == "res50":
        return driver_utils.HeavyNodeConfig(name="res50",
                                            input_type="floats",
                                            model_image="model-comp/pytorch-res50",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )

    elif model_name == "res18":
        return driver_utils.HeavyNodeConfig(name="res18",
                                            input_type="floats",
                                            model_image="model-comp/pytorch-res18",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )

    elif model_name == "res152":
        return driver_utils.HeavyNodeConfig(name="res152",
                                            input_type="floats",
                                            model_image="model-comp/pytorch-res152",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            )


def get_batch_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_batch_sizes = {}
    for h in hists:
        if "batch_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            mean = h[name]["mean"]
            mean_batch_sizes[model] = round(float(mean), 2)
    return mean_batch_sizes


def check_convergence(stats, config):
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
    window_size = 20
    p99_lats = stats["p99_lats"][-1*window_size:]
    mean_batch_size = np.mean([b[config.name] for b in stats["mean_batch_sizes"][-1*window_size:]])
    lr = linregress(x=range(len(p99_lats)), y=p99_lats)
    logger.info(lr)
    # pvalue checks against the null hypothesis that the
    # slope is 0. We are checking for the slope to be 0,
    # so we want to the null hypothesis to be true.
    if lr.pvalue < 0.1:
        return (False, np.sign(lr.slope))
    else:
        # Slope is 0, now check to see if mean batch_sizes are less than
        # configured batch size.
        if config.batch_size == 1.0:
            return (True, 0.0)
        elif mean_batch_size < config.batch_size:
            return (True, 0.0)
        else:
            logger.info("Slope is 0 but batch_sizes are too big")
            return (False, np.sign(1.))


class Predictor(object):

    def __init__(self, config, clipper_metrics):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "all_lats": [],
            "p99_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0
        self.config = config
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        self.get_clipper_metrics = clipper_metrics
        if self.get_clipper_metrics:
            self.stats["all_metrics"] = []
            self.stats["mean_batch_sizes"] = []

    def init_stats(self):
        self.latencies = []
        self.batch_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.batch_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["all_lats"].append(lats.tolist())
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        if self.get_clipper_metrics:
            metrics = self.cl.inspect_instance()
            batch_sizes = get_batch_sizes(metrics)
            self.stats["mean_batch_sizes"].append(batch_sizes)
            self.stats["all_metrics"].append(metrics)
            logger.info(("p99: {p99}, mean: {mean}, thruput: {thru}, "
                         "batch_sizes: {batches}").format(p99=p99, mean=mean, thru=thru,
                                                          batches=json.dumps(
                                                              batch_sizes, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))

    def predict(self, model_app_name, input_item):
        begin_time = datetime.now()

        def continuation(output):
            if output == DEFAULT_OUTPUT:
                return
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            queries_per_batch = max(400, 20*self.config.batch_size)
            if self.batch_num_complete % queries_per_batch == 0:
                self.print_stats()
                self.init_stats()

        return self.client.send_request(model_app_name, input_item).then(continuation)


class ModelBenchmarker(object):
    def __init__(self, config, queue, client_num):
        self.config = config
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        base_inputs = [np.array(np.random.rand(299*299*3), dtype=np.float32) for _ in range(1000)]
        self.inputs = [i for _ in range(60) for i in base_inputs]

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        # initialize delay to be very small
        self.delay = 0.001
        setup_clipper(model_config)
        time.sleep(5)
        predictor = Predictor(self.config, clipper_metrics=True)
        idx = 0
        while len(predictor.stats["thrus"]) < 5:
            predictor.predict(model_app_name=self.config.name, input_item=self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput - 0.0005
        logger.info("Initializing delay to {}".format(self.delay))

    def find_steady_state(self):
        setup_clipper(model_config)
        time.sleep(7)
        predictor = Predictor(self.config, clipper_metrics=True)
        idx = 0
        done = False
        # start checking for steady state after 7 trials
        last_checked_length = 7
        divergence_possible = True
        while not done:
            predictor.predict(model_app_name=self.config.name,
                              input_item=self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)
            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 5
                is_converged, slope_sign = check_convergence(predictor.stats,
                                                             self.config)
                # Diverging, try again with higher delay
                if (not is_converged) and divergence_possible and slope_sign > 0.0:
                    self.delay += 0.0005  # Increase by 500 us
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    return self.find_steady_state()
                elif is_converged:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.queue.put(predictor.stats)
                    return
                else:
                    logger.info("Not converged yet. Still waiting")
                    if slope_sign < 0.0:
                        divergence_possible = False

    def run(self):
        self.initialize_request_rate()
        self.find_steady_state()
        return


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--delay', type=float, help='inter-request delay')
    # parser.add_argument('-b', '--batch_size', type=int, help='batch_size')
    # parser.add_argument('-r', '--num_replicas', type=int, help='num_replicas')
    # parser.add_argument('-m', '--model', type=str,
    #                     help='model to benchmark. alexnet, res50, or res152')

    # args = parser.parse_args()

    # models = ["alexnet", "res50", "res152", "res18"]
    models = ["res152", "res18"]

    cpus = 1
    gpus = 1

    for m in models:
        for batch_size in [1, 2, 4, 6, 8, 10, 12, 16, 20, 30, 48]:
            model_config = get_heavy_node_config(
                model_name=m,
                batch_size=batch_size,
                num_replicas=1,
                cpus_per_replica=cpus,
                allocated_cpus=range(12, 21),
                allocated_gpus=range(gpus)
            )
            logger.info(json.dumps(model_config.__dict__))
            queue = Queue()
            benchmarker = ModelBenchmarker(model_config, queue=queue, client_num=0)

            p = Process(target=benchmarker.run, args=())
            p.start()
            all_stats = []
            all_stats.append(queue.get())
            p.join()

            cl = ClipperConnection(DockerContainerManager(redis_port=6380))
            cl.connect()
            driver_utils.save_results([model_config], cl,
                                    all_stats,
                                    "no-diverge-single_model_prof_{}".format(model_config.name))
