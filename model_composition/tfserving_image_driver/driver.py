import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
from io import BytesIO
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from multiprocessing import Process

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

########## Setup ##########

def setup_clipper(config):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="1-4")
    time.sleep(10)
    driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")
    return config


def get_heavy_node_config(batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus):

    return driver_utils.HeavyNodeConfig(name="inceptionv3-tfserve-compare",
                                        input_type="strings",
                                        model_image="model-comp/inceptionv3",
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True)




########## Benchmarking ##########

class Predictor(object):

    def __init__(self):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0

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
        self.stats["mean_lats"].append(mean)
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
            if self.batch_num_complete % 40 == 0:
                self.print_stats()
                self.init_stats()

        return self.client.send_request(model_app_name, input_item).then(continuation)

def gen_input():
    input_img = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    input_img = Image.fromarray(input_img.astype(np.uint8))
    inmem_inception_jpeg = BytesIO()
    resized_inception = input_img.resize((299,299)).convert('RGB')
    resized_inception.save(inmem_inception_jpeg, format="JPEG")
    inmem_inception_jpeg.seek(0)
    inception_input = inmem_inception_jpeg.read()
    return base64.b64encode(inception_input)

class ModelBenchmarker(object):
    def __init__(self, config):
        self.config = config

    def run(self):
        base_inputs = [gen_input() for _ in range(1000)]
        inputs = [i for _ in range(10) for i in base_inputs]

        start_time = datetime.now()
        predictor = Predictor()
        for input_item in inputs:
            predictor.predict(model_app_name=self.config.name, input_item=input_item)
            # time.sleep(0.005)
            time.sleep(0)
        while len(predictor.stats["thrus"]) < 8:
            time.sleep(1)

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        driver_utils.save_results([self.config], cl, predictor.stats, "single_model_prof_%s" % self.config.name)




if __name__ == "__main__":


    for cpus in [1, 2, 3]:
        for gpus in [0,]:
            for batch_size in [1, 2, 4, 6]:
                model_config = get_heavy_node_config(
                    batch_size=batch_size,
                    num_replicas=1,
                    cpus_per_replica=cpus,
                    allocated_cpus=range(6,16),
                    allocated_gpus=range(gpus)
                )
                logger.info("\nStarting trial: {}".format(
                    json.dumps(model_config.__dict__, indent=4)))

                setup_clipper(model_config)
                benchmarker = ModelBenchmarker(model_config)

                p = Process(target=benchmarker.run, args=())
                p.start()
                p.join()
