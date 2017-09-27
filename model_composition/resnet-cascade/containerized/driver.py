from __future__ import print_function
# import sys
# import os
import logging
import numpy as np
import time
from clipper_admin import ClipperConnection, DockerContainerManager
# from datetime import datetime
from multiprocessing import Process
from zmq_client import Client
from datetime import datetime
import argparse

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


def run(proc_num):
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
        time.sleep(0.01)


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
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, input):
        req_id = self.cur_req_id
        self.outstanding_reqs[req_id] = InflightReq()

        def res152_callback(response):
            self.outstanding_reqs[req_id].complete()
            self.latencies.append(self.outstanding_reqs[req_id].latency)
            self.num_complete += 1
            if self.num_complete % 100 == 0:
                self.print_stats()

            del self.outstanding_reqs[req_id]

        def res50_callback(response):
            if np.random.random() > 0.6:
                # logger.info("Requesting res152")
                self.client.send_request("res152", input, res152_callback)
            else:
                self.outstanding_reqs[req_id].complete()
                self.latencies.append(self.outstanding_reqs[req_id].latency)
                self.num_complete += 1
                if self.num_complete % 100 == 0:
                    self.print_stats()
                del self.outstanding_reqs[req_id]

        def alexnet_callback(response):
            # if np.random.random() > 0.7:
            if False:
                # logger.info("Requesting res50")
                self.client.send_request("res50", input, res50_callback)
            else:
                self.outstanding_reqs[req_id].complete()
                self.latencies.append(self.outstanding_reqs[req_id].latency)
                self.num_complete += 1
                if self.num_complete % 200 == 0:
                    self.print_stats()
                    self.init_stats()
                del self.outstanding_reqs[req_id]

        # logger.info("Requesting alexnet")
        self.client.send_request("alexnet", input, alexnet_callback)
        self.cur_req_id += 1


def setup_heavy_node(clipper_conn,
                     name,
                     input_type,
                     model_image,
                     allocated_cpus,
                     cpus_per_replica,
                     slo=500000,
                     num_replicas=1,
                     gpus=None,
                     batch_size=1):
    clipper_conn.register_application(name=name,
                                      default_output="TIMEOUT",
                                      slo_micros=slo,
                                      input_type=input_type)

    clipper_conn.deploy_model(name=name,
                              version=1,
                              image=model_image,
                              input_type=input_type,
                              num_replicas=num_replicas,
                              batch_size=batch_size,
                              gpus=gpus,
                              allocated_cpus=allocated_cpus,
                              cpus_per_replica=cpus_per_replica)

    clipper_conn.link_model_to_app(app_name=name, model_name=name)


def setup_clipper():
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.start_clipper(query_frontend_image="clipper/zmq_frontend:develop",
                     redis_cpu_str="0",
                     mgmt_cpu_str="8",
                     query_cpu_str="1-5,9-13")
    time.sleep(10)
    setup_heavy_node(cl, "alexnet", "floats", "model-comp/pytorch-alexnet",
                     allocated_cpus=range(16, 32),
                     cpus_per_replica=1,
                     gpus=[0],
                     batch_size=4,
                     num_replicas=1,
                     )
    # setup_heavy_node(cl, "res50", "floats", "model-comp/pytorch-res50", gpus=[1])
    # setup_heavy_node(cl, "res152", "floats", "model-comp/pytorch-res152", gpus=[2])
    time.sleep(10)
    logger.info("Clipper is set up")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Resnet-Cascade-Driver')
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--num_procs', type=int, default=1)
    args = parser.parse_args()
    if args.setup:
        setup_clipper()
    if args.run:
        processes = []
        for i in range(args.num_procs):
            p = Process(target=run, args=('%d'.format(i),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
