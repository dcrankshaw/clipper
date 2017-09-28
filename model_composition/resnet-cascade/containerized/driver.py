from __future__ import print_function
# import sys
# import os
import logging
import numpy as np
import time
from clipper_admin import ClipperConnection, DockerContainerManager
# from datetime import datetime
from multiprocessing import Process, Queue
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from datetime import datetime
import argparse
import json

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)


DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4


def run(proc_num, results_queue):
    height, width = 299, 299
    channels = 3
    logger.info("Generating random inputs")
    xs = [np.random.random((height, width, channels)).flatten().astype(np.float32)
          for _ in range(10000)]
    logger.info("Starting predictions")
    predictor = Predictor()
    for x in xs:
        # x = np.random.random((height, width, channels)).flatten().astype(np.float32)
        # logger.info("sending prediction")
        predictor.predict(x)
        time.sleep(0.005)
    # let the experiment run for 15 more seconds
    time.sleep(15)
    results_queue.put(predictor.stats)


class InflightReq(object):

    def __init__(self):
        self.start_time = datetime.now()

    def complete(self):
        self.latency = (datetime.now() - self.start_time).total_seconds()
        # logger.info("Completed in {} seconds".format(self.latency))


class Predictor(object):

    def __init__(self):
        self.outstanding_reqs = {}
        self.client = Client("localhost", 4456, 4455)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []}

    def init_stats(self):
        self.latencies = []
        self.num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, input):
        req_id = self.cur_req_id
        self.outstanding_reqs[req_id] = InflightReq()

        def complete_req(response):
            self.outstanding_reqs[req_id].complete()
            self.latencies.append(self.outstanding_reqs[req_id].latency)
            self.num_complete += 1
            if self.num_complete % 200 == 0:
                self.print_stats()
                self.init_stats()
            del self.outstanding_reqs[req_id]

        def res50_callback(response):
            if np.random.random() > 0.6:
                # logger.info("Requesting res152")
                self.client.send_request("res152", input).then(complete_req)
            else:
                complete_req(response)

        def alexnet_completion(response):
            # if np.random.random() > 0.7:
            if False:
                self.client.send_request("res50", input).then(res50_callback)
            else:
                complete_req(response)

        alexnet_future = self.client.send_request("alexnet", input)
        alexnet_future.then(alexnet_completion)
        self.cur_req_id += 1


def setup_clipper(alexnet_config):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(query_frontend_image="clipper/zmq_frontend:develop",
                     redis_cpu_str="0",
                     mgmt_cpu_str="8",
                     query_cpu_str="1-5,9-13")
    time.sleep(10)
    driver_utils.setup_heavy_node(cl, alexnet_config)
    # setup_heavy_node(cl, "res50", "floats", "model-comp/pytorch-res50", gpus=[1])
    # setup_heavy_node(cl, "res152", "floats", "model-comp/pytorch-res152", gpus=[2])
    time.sleep(10)
    logger.info("Clipper is set up")
    return cl, [alexnet_config]


def run_experiment(num_clients, config, experiment_name):
    cl, configs = setup_clipper(config)
    client_metrics_queue = Queue()
    processes = []
    for i in range(num_clients):
        p = Process(target=run, args=('%d'.format(i), client_metrics_queue))
        p.start()
        processes.append(p)

    client_metrics = []
    for p in processes:
        p.join()
        client_metrics.append(client_metrics_queue.get())

    driver_utils.save_results(configs, cl, client_metrics, experiment_name)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Resnet-Cascade-Driver')
    # parser.add_argument('--setup', action='store_true')
    # parser.add_argument('--run', action='store_true')
    # parser.add_argument('--num_clients', type=int, default=1)
    # args = parser.parse_args()

    num_clients = 3

    for num_reps in range(1, 5):
        for batch_size in [1, 2, 4, 8, 16, 32]:
            alexnet_config = driver_utils.HeavyNodeConfig("alexnet",
                                                          "floats",
                                                          "model-comp/pytorch-alexnet",
                                                          allocated_cpus=range(16, 32),
                                                          cpus_per_replica=1,
                                                          gpus=range(num_reps),
                                                          batch_size=batch_size,
                                                          num_replicas=num_reps)
            run_experiment(num_clients, alexnet_config, "replication_and_batch_size_take_2")
