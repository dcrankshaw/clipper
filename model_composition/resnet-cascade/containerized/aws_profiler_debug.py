import sys
# import os
# import argparse
import numpy as np
import time
# import base64
import logging
import json

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
# from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
# from containerized_utils.driver_utils import (INCREASING, DECREASING,
#                                               CONVERGED_HIGH, CONVERGED,
#                                               UNKNOWN, CONVERGED_LOW)
from multiprocessing import Process, Queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"

CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"


"""
Models:
    + Driver 1:
        + inception
        + resnet 152
        + logreg
        + kernel svm

"""


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


def setup_clipper(config):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="0,16,1,17,2,18,3,19")
    time.sleep(10)
    driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")
    return CLIPPER_ADDRESS


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
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
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
                         "batch_sizes: {batches}, queue_sizes: {queues}").format(
                             p99=p99,
                             mean=mean,
                             thru=thru,
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

            # trial_length = max(1000, 25 * self.batch_size)
            trial_length = 4000
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
        self.initialize_request_rate_queue_drain()
        self.find_steady_state()
        # self.process_queue()
        return

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        # initialize delay to be very small
        self.delay = 0.001
        self.queries_per_sleep = 1
        self.clipper_address = setup_clipper(self.config)
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        time.sleep(30)
        predictor = Predictor(self.clipper_address, clipper_metrics=True,
                              batch_size=self.max_batch_size)
        idx = 0

        # First warm up the model.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet
        while len(predictor.stats["thrus"]) < 10:
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
        self.cl.drain_queues()
        time.sleep(10)



        # predictor = Predictor(self.clipper_address, clipper_metrics=True,
        #                       batch_size=self.max_batch_size)
        #
        # # Now initialize request rate
        # while len(predictor.stats["thrus"]) < 30:
        #     predictor.predict(self.config.name, self.inputs[idx])
        #     idx += 1
        #     if idx % self.queries_per_sleep == 0:
        #         time.sleep(self.delay)
        #     idx = idx % len(self.inputs)
        #
        # max_thruput = np.mean(predictor.stats["thrus"][5:])
        # # # USE MEDIAN AS THROUGHPUT ESTIMATOR
        # # max_thruput = np.percentile(predictor.stats["thrus"][5:], 50)
        # self.queue.put((self.clipper_address, predictor.stats))
        # self.delay = 1.0 / max_thruput
        # # if self.delay < 0.01:
        # #     self.queries_per_sleep = 2
        # #     self.delay = self.delay*2.0 - 0.001
        # logger.info("Initializing delay to {}".format(self.delay))
        # predictor.client.stop()
        # logger.info("ZMQ client stopped")
        # del predictor
        # self.cl.drain_queues()
        # time.sleep(10)
        # self.cl.drain_queues()

    def increase_delay(self, multiple=1.0):
        if self.delay < 0.01:
            self.delay += 0.0001*multiple
        elif self.delay < 0.02:
            self.delay += 0.0002*multiple
        else:
            self.delay += 0.001*multiple

    def decrease_delay(self):
        self.increase_delay(multiple=-0.5)

    def initialize_request_rate_queue_drain(self):
        # initialize delay to be very small
        self.delay = 0.001
        self.queries_per_sleep = 1
        self.clipper_address = setup_clipper(self.config)
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        time.sleep(30)
        predictor = Predictor(self.clipper_address, clipper_metrics=True,
                              batch_size=self.max_batch_size)
        idx = 0

        # First warm up the model.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet
        while len(predictor.stats["thrus"]) < 10:
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
        self.cl.drain_queues()
        time.sleep(10)
        predictor = Predictor(self.clipper_address, clipper_metrics=True,
                              batch_size=self.max_batch_size)

        idx = 0
        while len(predictor.stats["mean_queue_sizes"]) == 0 or predictor.stats["mean_queue_sizes"][-1][self.config.name] < 80000:
            predictor.predict(self.config.name, self.inputs[idx])
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)
        logger.info("Queue is big. Stopping sending queries")
        while predictor.stats["mean_queue_sizes"][-1][self.config.name] > 10000:
            time.sleep(1)

        queue_sizes = [q[self.config.name] for q in predictor.stats["mean_queue_sizes"]]
        diffs = np.diff(queue_sizes)
        first_decrease_idx = 0
        for i, d in enumerate(diffs):
            if d < 0:
                first_decrease_idx = i + 1  # add 1 because np.diff does the gaps between numbers
                break
        assert first_decrease_idx > 1
        logger.info("First decrease idx: {}".format(first_decrease_idx))
        logger.info("Included thrus: {}".format(predictor.stats["thrus"][first_decrease_idx:]))
        max_thruput = np.mean(predictor.stats["thrus"][first_decrease_idx:])
        self.queue.put((self.clipper_address, predictor.stats))
        self.delay = 1.0 / max_thruput
        # if self.delay < 0.01:
        #     self.queries_per_sleep = 2
        #     self.delay = self.delay*2.0 - 0.001
        logger.info("Initializing delay to {}".format(self.delay))
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor
        self.cl.drain_queues()
        time.sleep(10)
        self.cl.drain_queues()
        return

    def process_queue(self):
        self.delay = 0.001
        self.cl.drain_queues()
        time.sleep(10)
        logger.info("Queue is drained")
        predictor = Predictor(self.clipper_address, clipper_metrics=True,
                              batch_size=self.max_batch_size)

        idx = 0
        while len(predictor.stats["mean_queue_sizes"]) == 0 or predictor.stats["mean_queue_sizes"][-1][self.config.name] < 80000:
            predictor.predict(self.config.name, self.inputs[idx])
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)
        logger.info("Queue is big. Stopping sending queries")
        while predictor.stats["mean_queue_sizes"][-1][self.config.name] > 10000:
            time.sleep(1)
        self.queue.put((self.clipper_address, predictor.stats))
        return

    def find_steady_state(self):
        # self.cl.cm.reset()
        self.cl.drain_queues()
        time.sleep(10)
        logger.info("Queue is drained")
        predictor = Predictor(self.clipper_address, clipper_metrics=True,
                              batch_size=self.max_batch_size)

        idx = 0
        while len(predictor.stats["thrus"]) < 30:
            predictor.predict(self.config.name, self.inputs[idx])
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)
        convergence_state = driver_utils.check_convergence_via_queue(
            predictor.stats, [self.config, ])
        logger.info("Convergence state: {}".format(convergence_state))
        self.queue.put((self.clipper_address, predictor.stats))
        return

        #
        #
        # done = False
        # # start checking for steady state after 10 trials
        # last_checked_length = 40
        #
        # while not done:
        #     predictor.predict(self.config.name, self.inputs[idx])
        #     if idx % self.queries_per_sleep == 0:
        #         time.sleep(self.delay)
        #     idx += 1
        #     idx = idx % len(self.inputs)
        #
        #     if len(predictor.stats["thrus"]) > last_checked_length:
        #         last_checked_length = len(predictor.stats["thrus"]) + 1
        #         convergence_state = driver_utils.check_convergence_via_queue(
        #             predictor.stats, [self.config, ])
        #         # Diverging, try again with higher
        #         # delay
        #         if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
        #             self.increase_delay()
        #             logger.info("Increasing delay to {}".format(self.delay))
        #             done = True
        #             del predictor
        #             return self.find_steady_state()
        #         elif convergence_state == CONVERGED:
        #             logger.info("Converged with delay of {}".format(self.delay))
        #             done = True
        #             self.queue.put((self.clipper_address, predictor.stats))
        #             return
        #         elif len(predictor.stats) > 40:
        #             self.increase_delay()
        #             logger.info("Increasing delay to {}".format(self.delay))
        #             done = True
        #             del predictor
        #             return self.find_steady_state()
        #         elif convergence_state == DECREASING or convergence_state == UNKNOWN:
        #             logger.info("Not converged yet. Still waiting")
        #         elif convergence_state == CONVERGED_LOW:
        #             logger.info("Converged with too low batch sizes.")
        #             done = True
        #             self.queue.put((self.clipper_address, predictor.stats))
        #             return
        #
        #             self.decrease_delay()
        #             logger.info(("Converged with too low batch sizes. "
        #                         "Decreasing delay to {}").format(self.delay))
        #             done = True
        #             del predictor
        #             return self.find_steady_state()
        #         else:
        #             logger.error("Unknown convergence state: {}".format(convergence_state))
        #             sys.exit(1)


if __name__ == "__main__":

    queue = Queue()

    for batch_size in [8, 16, 32, 4, 64, 1, 2]:
        config = get_heavy_node_config(
            model_name=RES50,
            batch_size=batch_size,
            num_replicas=1,
            cpus_per_replica=1,
            allocated_cpus=range(4, 8),
            allocated_gpus=range(4)
        )
        client_num = 0
        benchmarker = DriverBenchmarker(config, queue, client_num, 0.2*batch_size)

        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        _, init_stats = queue.get()
        clipper_address, steady_stats = queue.get()
        all_stats.append(steady_stats)

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        loop_durs = []
        with open("/home/ubuntu/logs/loop_duration.log", "r") as f:
            for line in f:
                loop_durs.append(float(line.strip()))
        handle_durs = []
        with open("/home/ubuntu/logs/handle_duration.log", "r") as f:
            for line in f:
                handle_durs.append(float(line.strip()))
        recv_times = []
        with open("/home/ubuntu/logs/recv_times.log", "r") as f:
            for line in f:
                recv_times.append(float(line.strip()))
        container_metrics = {
            "loop_durs": loop_durs,
            "handle_durs": handle_durs,
            "recv_times": recv_times,
        }

        fname = "results-batch-{batch}".format(batch=batch_size)
        driver_utils.save_results([config, ],
                                  all_stats,
                                  [init_stats, ],
                                  # [],
                                  "pytorch_res50_smp_aws_drain_queue_then_actual_profile",
                                  prefix=fname,
                                  container_metrics=container_metrics)
        cl.stop_all()

    sys.exit(0)
