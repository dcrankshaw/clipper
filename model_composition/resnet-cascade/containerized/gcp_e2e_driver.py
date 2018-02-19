import sys
# import os
# import argparse
import numpy as np
import time
# import base64
import logging
import json

from clipper_admin import ClipperConnection, GCPContainerManager
from datetime import datetime
# from PIL import Image
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

GCP_CLUSTER_NAME = "e2e-cascade"


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


def setup_clipper_gcp(configs):
    cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
    cl.stop_all()
    cl.start_clipper()
    time.sleep(30)
    for c in configs:
        driver_utils.setup_heavy_node_gcp(cl, c, DEFAULT_OUTPUT)
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

    def __init__(self, clipper_address, clipper_metrics):
        self.outstanding_reqs = {}
        self.client = Client(clipper_address, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "all_lats": [],
            "mean_lats": [],
            "node_thrus": {"alex": [], "res50": [], "res152": []},
        }
        self.total_num_complete = 0
        self.cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        self.cl.connect()
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
        self.node_counts = {"alex": 0, "res50": 0, "res152": 0}

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.batch_num_complete) / (end_time - self.start_time).total_seconds()
        cur_batch_node_thrus = {}
        for n in self.node_counts:
            node_thru = float(self.node_counts[n]) / (end_time - self.start_time).total_seconds()
            self.stats["node_thrus"][n].append(node_thru)
            cur_batch_node_thrus[n] = node_thru
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
            logger.info(("p99: {p99}, mean: {mean}, e2e thruput: {thru}, "
                         "batch sizes: {batches}, queue sizes: {queues}, "
                         "node thrus: {node_thrus}").format(
                             p99=p99, mean=mean, thru=thru,
                             batches=json.dumps(
                                 batch_sizes, sort_keys=True),
                             queues=json.dumps(
                                 queue_sizes, sort_keys=True),
                             node_thrus=json.dumps(
                                 cur_batch_node_thrus, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))

    def predict(self, input_item):
        begin_time = datetime.now()

        def complete():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            if self.batch_num_complete % 700 == 0:
                self.print_stats()
                self.init_stats()

        def res152_cont(output):
            try:
                if output == DEFAULT_OUTPUT:
                    return
                else:
                    self.node_counts["res152"] += 1
                    complete()
            except Exception as e:
                print(e)

        def res50_cont(output):
            try:
                if output == DEFAULT_OUTPUT:
                    return
                else:
                    self.node_counts["res50"] += 1
                    idk = np.random.random() > 0.4633
                    # idk = True
                    if idk:
                        self.client.send_request(RES152, input_item).then(res152_cont)
                    else:
                        complete()
            except Exception as e:
                print(e)

        def alex_cont(output):
            try:
                if output == DEFAULT_OUTPUT:
                    return
                else:
                    self.node_counts["alex"] += 1
                    idk = np.random.random() > 0.192
                    # idk = True
                    if idk:
                        self.client.send_request(RES50, input_item).then(res50_cont)
                    else:
                        complete()
            except Exception as e:
                print(e)

        return self.client.send_request(ALEXNET, input_item).then(alex_cont)


class DriverBenchmarker(object):
    def __init__(self, configs, queue, client_num, convergence_metric, latency_upper_bound=None):
        self.configs = configs
        self.convergence_metric = convergence_metric
        self.latency_upper_bound = latency_upper_bound
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        self.inputs = [np.array(np.random.rand(299*299*3), dtype=np.float32) for _ in range(1000)]

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
        self.clipper_address = setup_clipper_gcp(self.configs)
        self.cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        self.cl.connect()
        time.sleep(30)
        predictor = Predictor(self.clipper_address, clipper_metrics=True)
        idx = 0

        # First warm up the models.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet
        while len(predictor.stats["thrus"]) < 30:
            predictor.predict(self.inputs[idx])
            idx += 1
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx = idx % len(self.inputs)

        # Now let the queue drain
        logger.info("Draining queues")

        self.cl.drain_queues()
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor
        time.sleep(10)
        self.cl.drain_queues()
        predictor = Predictor(self.clipper_address, clipper_metrics=True)

        # Now initialize request rate
        while len(predictor.stats["thrus"]) < 12:
            predictor.predict(self.inputs[idx])
            idx += 1
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][2:])
        self.delay = 1.0 / max_thruput
        # if self.delay < 0.01:
        #     self.queries_per_sleep = 2
        #     self.delay = self.delay*2.0 - 0.001
        logger.info("Initializing delay to {}".format(self.delay))
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor

    def increase_delay(self, multiple=1.0):
        if self.delay < 0.01:
            self.delay += 0.0001*multiple
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
        self.cl.drain_queues()
        logger.info("Queue is drained")
        predictor = Predictor(self.clipper_address, clipper_metrics=True)
        self.active = False
        while not self.active:
            logger.info("Trying to connect to Clipper")

            def callback(output):
                if output == DEFAULT_OUTPUT:
                    return
                else:
                    logger.info("Succesful query issued")
                    self.active = True
            predictor.client.send_request(RES152, self.inputs[0]).then(callback)
            time.sleep(1)

        idx = 0
        done = False
        # start checking for steady state after 10 trials
        last_checked_length = 10
        while not done:
            predictor.predict(self.inputs[idx])
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 1
                if self.convergence_metric == "queue":
                    convergence_state = driver_utils.check_convergence_via_queue(predictor.stats,
                                                                                 self.configs)
                elif self.convergence_metric == "latency":
                    convergence_state = driver_utils.check_convergence(predictor.stats,
                                                                       self.configs,
                                                                       self.latency_upper_bound)
                else:
                    logger.error("{} is invalid convergence metric".format(self.convergence_metric))
                    assert False
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    predictor.client.stop()
                    del predictor
                    self.cl.drain_queues()
                    time.sleep(10)
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
                    predictor.client.stop()
                    del predictor
                    self.cl.drain_queues()
                    time.sleep(10)
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                elif convergence_state == CONVERGED_LOW:
                    logger.info("Converged with too low batch sizes.")
                    done = True
                    self.queue.put((self.clipper_address, predictor.stats))
                    return
                    # self.decrease_delay()
                    # logger.info(("Converged with too low batch sizes. "
                    #              "Decreasing delay to {}").format(self.delay))
                    # done = True
                    # del predictor
                    # self.cl.drain_queues()
                    # time.sleep(10)
                    # return self.find_steady_state()
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)


if __name__ == "__main__":
    # cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
    # cl.connect()
    # cl.stop_all()
    # sys.exit(0)

    queue = Queue()
    alex_batch = 8
    res50_batch = 4
    res152_batch = 2

    latency_upper_bound = None
    convergence_metric = "latency"

    reps = [
        # (1,1,1),
        # (1,1,2),
        (2, 2, 2),
            ]

    for alex_reps, res50_reps, res152_reps in reps:
        configs = [
            setup_alexnet(alex_batch, alex_reps, 1, "k80"),
            setup_res50(res50_batch, res50_reps, 1, "p100"),
            setup_res152(res152_batch, res152_reps, 1, "p100")
        ]
        client_num = 0
        benchmarker = DriverBenchmarker(configs,
                                        queue,
                                        client_num,
                                        convergence_metric,
                                        latency_upper_bound)

        p = Process(target=benchmarker.run)
        p.start()
        # benchmarker.run()

        all_stats = []
        clipper_address, stats = queue.get()
        all_stats.append(stats)

        cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        cl.connect()

        fname = "alex_{}-r50_{}-r152_{}_{}-convergence".format(
            alex_reps, res50_reps, res152_reps, convergence_metric)
        driver_utils.save_results(configs,
                                  cl,
                                  all_stats,
                                  "cascade_e2e-DEBUG",
                                  prefix=fname)
        cl.stop_all()

    sys.exit(0)
