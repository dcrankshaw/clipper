import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json

from clipper_admin import ClipperConnection, DockerContainerManager, GCPContainerManager
from datetime import datetime
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from containerized_utils.driver_utils import (INCREASING, DECREASING,
                                              CONVERGED_HIGH, CONVERGED,
                                              UNKNOWN, CONVERGED_LOW)
from multiprocessing import Process, Queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"

CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"

GCP_CLUSTER_NAME = "single-model-profiles-cascade"


"""
Models:
    + Driver 1:
        + inception
        + resnet 152
        + logreg
        + kernel svm

"""

def setup_alexnet(batch_size,
                  num_replicas,
                  cpus_per_replica,
                  gpu_type):

    image = "gcr.io/clipper-model-comp/pytorch-alexnet:bench"
    return driver_utils.HeavyNodeConfigGCP(name=ALEXNET,
                                           input_type="floats",
                                           model_image=image,
                                           cpus_per_replica=cpus_per_replica,
                                           gpu_type=gpu_type,
                                           batch_size=batch_size,
                                           num_replicas=num_replicas,
                                           no_diverge=True)

def setup_res50(batch_size,
                  num_replicas,
                  cpus_per_replica,
                  gpu_type):

    image = "gcr.io/clipper-model-comp/pytorch-res50:bench"
    return driver_utils.HeavyNodeConfigGCP(name=RES50,
                                           input_type="floats",
                                           model_image=image,
                                           cpus_per_replica=cpus_per_replica,
                                           gpu_type=gpu_type,
                                           batch_size=batch_size,
                                           num_replicas=num_replicas,
                                           no_diverge=True)

def setup_res152(batch_size,
                  num_replicas,
                  cpus_per_replica,
                  gpu_type):

    image = "gcr.io/clipper-model-comp/pytorch-res152:bench"
    return driver_utils.HeavyNodeConfigGCP(name=RES152,
                                           input_type="floats",
                                           model_image=image,
                                           cpus_per_replica=cpus_per_replica,
                                           gpu_type=gpu_type,
                                           batch_size=batch_size,
                                           num_replicas=num_replicas,
                                           no_diverge=True)

def setup_clipper_gcp(config):
    cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
    cl.stop_all()
    cl.start_clipper()
    time.sleep(30)
    driver_utils.setup_heavy_node_gcp(cl, config, DEFAULT_OUTPUT)
    time.sleep(10)
    clipper_address = cl.cm.query_frontend_internal_ip
    logger.info("Clipper is set up on {}".format(clipper_address))
    return clipper_address

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

def get_queue_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_queue_sizes = {}
    for h in hists:
        if "queue_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = h[name]["mean"]
            mean_queue_sizes[model] = round(float(mean), 2)
    return mean_queue_sizes

class Predictor(object):

    def __init__(self, clipper_address, clipper_metrics, batch_size):
        self.outstanding_reqs = {}
        self.client = Client(clipper_address, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "all_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0
        self.cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        self.cl.connect()
        self.batch_size = batch_size
        self.get_clipper_metrics = clipper_metrics
        if self.get_clipper_metrics:
            self.stats["all_metrics"] = []
            self.stats["mean_batch_sizes"] = []
            self.stats["mean_queue_sizes"] = []

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
        self.stats["p99_lats"].append(p99)
        self.stats["all_lats"].append(self.latencies)
        self.stats["mean_lats"].append(mean)
        if self.get_clipper_metrics:
            metrics = self.cl.inspect_instance()
            batch_sizes = get_batch_sizes(metrics)
            queue_sizes = get_queue_sizes(metrics)
            self.stats["mean_batch_sizes"].append(batch_sizes)
            self.stats["mean_queue_sizes"].append(queue_sizes)
            self.stats["all_metrics"].append(metrics)
            logger.info(("p99: {p99}, mean: {mean}, thruput: {thru}, "
                         "batch_sizes: {batches}, queue_sizes: {queues}").format(p99=p99, mean=mean, thru=thru,
                                                          batches=json.dumps(
                                                              batch_sizes, sort_keys=True),
                                                          queues=json.dumps(
                                                              queue_sizes, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))

    def predict(self, model_app_name, input_item):
        begin_time = datetime.now()

        def on_complete(output):
            if output == DEFAULT_OUTPUT:
                return
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1

            trial_length = max(400, 25 * self.batch_size)
            if self.batch_num_complete % trial_length == 0:
                self.print_stats()
                self.init_stats()

        return self.client.send_request(model_app_name, input_item).then(on_complete)


class DriverBenchmarker(object):
    def __init__(self, config, queue, client_num, latency_upper_bound):
        self.config = config
        self.max_batch_size = config.batch_size
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        self.inputs = [np.array(np.random.rand(299*299*3), dtype=np.float32) for _ in range(1000)]
        # self.input_generator_fn = self._get_input_generator_fn(model_app_name=self.config.name)
        # base_inputs = [self.input_generator_fn() for _ in range(1000)]
        # self.inputs = base_inputs
        # self.inputs = [i for _ in range(5) for i in base_inputs]
        self.latency_upper_bound = latency_upper_bound

    def run(self):
        self.initialize_request_rate()
        self.find_steady_state()
        return

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        # initialize delay to be very small
        self.delay = 0.001
        self.queries_per_sleep = 1
        self.clipper_address = setup_clipper_gcp(self.config)
        self.cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        self.cl.connect()
        time.sleep(30)
        predictor = Predictor(self.clipper_address, clipper_metrics=True, batch_size=self.max_batch_size)
        idx = 0

        # First warm up the model.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet
        while len(predictor.stats["thrus"]) < 27:
            predictor.predict(self.config.name, self.inputs[idx])
            idx += 1
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx = idx % len(self.inputs)

        # Now let the queue drain
        logger.info("Draining queue")

        self.cl.drain_queues()
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor
        time.sleep(10)
        predictor = Predictor(self.clipper_address, clipper_metrics=True, batch_size=self.max_batch_size)
        # while predictor.stats["mean_queue_sizes"][-1] > 0:
        #     sleep_time_secs = 5
        #     logger.info("Queue has {q_len} queries. Sleeping {sleep}".format(
        #         q_len=predictor.stats["mean_queue_sizes"][-1],
        #         sleep=sleep_time_secs))
        #     time.sleep(sleep_time_secs)

        # Now initialize request rate
        while len(predictor.stats["thrus"]) < 10:
            predictor.predict(self.config.name, self.inputs[idx])
            idx += 1
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        if self.delay < 0.01:
            self.queries_per_sleep = 2
            self.delay = self.delay*2.0 - 0.001
        logger.info("Initializing delay to {}".format(self.delay))
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor

    def increase_delay(self, multiple=1.0):
        if self.delay < 0.01:
            self.delay += 0.00005*multiple
        elif self.delay < 0.02:
            self.delay += 0.0002*multiple
        else:
            self.delay += 0.001*multiple

    def decrease_delay(self):
        self.increase_delay(multiple=-0.5)

    def find_steady_state(self):
        # self.cl.cm.reset()
        self.cl.drain_queues()
        time.sleep(10)
        logger.info("Queue is drained")
        predictor = Predictor(self.clipper_address, clipper_metrics=True, batch_size=self.max_batch_size)
        self.active = False
        while not self.active:
            logger.info("Trying to connect to Clipper")
            def callback(output):
                if output == DEFAULT_OUTPUT:
                    return
                else:
                    logger.info("Succesful query issued")
                    self.active = True
            predictor.client.send_request(self.config.name, self.inputs[0]).then(callback)
            time.sleep(1)

        idx = 0
        done = False
        # start checking for steady state after 10 trials
        last_checked_length = 10
        while not done:
            predictor.predict(self.config.name, self.inputs[idx])
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 1
                convergence_state = driver_utils.check_convergence_via_queue(
                    predictor.stats, [self.config,], self.latency_upper_bound)
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    del predictor
                    return self.find_steady_state()
                elif convergence_state == CONVERGED:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.queue.put((self.clipper_address, predictor.stats))
                    return
                elif len(predictor.stats) > 40:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    del predictor
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                elif convergence_state == CONVERGED_LOW:
                    self.decrease_delay()
                    logger.info("Converged with too low batch sizes. Decreasing delay to {}".format(self.delay))
                    done = True
                    del predictor
                    return self.find_steady_state()
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)


if __name__ == "__main__":

    queue = Queue()
    for gpu_type in ["p100", "k80"]:
        for num_cpus in [2, 1, 4]:
            for batch_size in [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96]:
                if gpu_type == "k80" and batch_size > 32:
                    continue
                config = setup_res50(batch_size, 1, num_cpus, gpu_type)
                client_num = 0
                benchmarker = DriverBenchmarker(config, queue, client_num, 0.2*batch_size)

                p = Process(target=benchmarker.run)
                p.start()

                all_stats = []
                clipper_address, stats = queue.get()
                all_stats.append(stats)

                cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
                cl.connect()

                fname = "results-{gpu}-{num_cpus}-{batch}".format(gpu=gpu_type, num_cpus=num_cpus, batch=batch_size)
                driver_utils.save_results([config,], cl, all_stats, "pytorch_res50_smp_gcp_queue_convergence", prefix=fname)

    for gpu_type in ["p100", "k80"]:
        for num_cpus in [2, 1, 4]:
            for batch_size in [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96]:
                if gpu_type == "k80" and batch_size > 32:
                    continue
                config = setup_res152(batch_size, 1, num_cpus, gpu_type)
                client_num = 0
                benchmarker = DriverBenchmarker(config, queue, client_num, 0.2*batch_size)

                p = Process(target=benchmarker.run)
                p.start()

                all_stats = []
                clipper_address, stats = queue.get()
                all_stats.append(stats)

                cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
                cl.connect()

                fname = "results-{gpu}-{num_cpus}-{batch}".format(gpu=gpu_type, num_cpus=num_cpus, batch=batch_size)
                driver_utils.save_results([config,], cl, all_stats, "pytorch_res152_smp_gcp_queue_convergence", prefix=fname)

    for gpu_type in ["p100", "k80"]:
        for num_cpus in [2, 1, 4]:
            for batch_size in [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96]:
                if gpu_type == "k80" and batch_size > 32:
                    continue
                config = setup_alexnet(batch_size, 1, num_cpus, gpu_type)
                client_num = 0
                benchmarker = DriverBenchmarker(config, queue, client_num, 0.2*batch_size)

                p = Process(target=benchmarker.run)
                p.start()

                all_stats = []
                clipper_address, stats = queue.get()
                all_stats.append(stats)

                cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
                cl.connect()

                fname = "results-{gpu}-{num_cpus}-{batch}".format(gpu=gpu_type, num_cpus=num_cpus, batch=batch_size)
                driver_utils.save_results([config,], cl, all_stats, "pytorch_alexnet_smp_gcp_queue_convergence", prefix=fname)

    sys.exit(0)
