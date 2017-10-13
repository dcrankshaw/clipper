from __future__ import print_function
# import sys
# import argparse
import os
import logging
import numpy as np
import time
import math

from clipper_admin import ClipperConnection, DockerContainerManager
from multiprocessing import Process, Queue
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from datetime import datetime

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
AUTOCOMPLETION_MODEL_APP_NAME = "autocompletion"
LSTM_MODEL_APP_NAME = "lstm"
NMT_MODEL_APP_NAME = "nmt"

AUTOCOMPLETION_IMAGE_NAME = "model-comp/tf-autocomplete"
LSTM_IMAGE_NAME = "model-comp/theano-lstm"
NMT_IMAGE_NAME = "model-comp/nmt"

VALID_MODEL_NAMES = [
    AUTOCOMPLETION_MODEL_APP_NAME,
    LSTM_MODEL_APP_NAME,
    NMT_MODEL_APP_NAME
]

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


def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus,
                          input_size):

    if model_name == LSTM_MODEL_APP_NAME:
        return driver_utils.HeavyNodeConfig(name=LSTM_MODEL_APP_NAME,
                                            input_type="strings",
                                            model_image=LSTM_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            input_size=input_size,
                                            )

    elif model_name == NMT_MODEL_APP_NAME:
        return driver_utils.HeavyNodeConfig(name=NMT_MODEL_APP_NAME,
                                            input_type="strings",
                                            model_image=NMT_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            input_size=input_size,
                                            )

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
            if self.batch_num_complete % 50 == 0:
                self.print_stats()
                self.init_stats()

        return self.client.send_request(model_app_name, input_item).then(continuation)


class ModelBenchmarker(object):
    def __init__(self, config, queue):
        self.config = config
        self.loaded_reviews = False
        self.loaded_german = False
        self.queue = queue

    def run(self):
        logger.info("Generating random inputs")
        inputs_generator_fn = self._get_inputs_generator_fn(self.config.name)
        inputs = inputs_generator_fn()
        logger.info("Starting predictions")
        # start_time = datetime.now()
        predictor = Predictor()
        for input_item in inputs:
            predictor.predict(model_app_name=self.config.name, input_item=input_item)
            time.sleep(0.005)

        while len(predictor.stats["thrus"]) < 15:
            time.sleep(1)

        self.queue.put(predictor.stats)

    def _gen_reviews_inputs(self, num_inputs=5000, input_length=200):
        if not self.loaded_reviews:
            self.reviews = self._load_reviews()
            self.loaded_reviews = True

        reviews_len = len(self.reviews)
        inputs = []
        for _ in range(num_inputs):
            review_idx = np.random.randint(reviews_len)
            review = self.reviews[review_idx]
            # Keep the first 200 words of the review,
            # or extend the review to exactly 200 words
            if len(review) < input_length:
                expansion_factor = int(math.ceil(float(input_length)/len(review)))
                for i in range(expansion_factor):
                    review = review + " " + review
            review = review[:input_length]
            inputs.append(review)
        return inputs

    def _gen_german_inputs(self, num_inputs=5000, input_length=10):
        if not self.loaded_german:
            self.german_text = self._load_german()
            self.loaded_german = True

        inputs = []
        num_gen_inputs = 0
        while num_gen_inputs < num_inputs:
            idx = np.random.randint(len(self.german_text))
            text = self.german_text[idx]
            words = text.split()
            if len(words) > input_length:
                words = words[:input_length]
                inputs.append(" ".join(words))
                num_gen_inputs += 1

        return inputs

    def _get_inputs_generator_fn(self, model_name):
        if model_name == AUTOCOMPLETION_MODEL_APP_NAME:
            return lambda: self._gen_reviews_inputs(num_inputs=5000, input_length=10)
        elif model_name == LSTM_MODEL_APP_NAME:
            return lambda: self._gen_reviews_inputs(num_inputs=5000,
                                                    input_length=self.config.input_size)
        elif model_name == NMT_MODEL_APP_NAME:
            return lambda: self._gen_german_inputs(num_inputs=5000,
                                                   input_length=self.config.input_size)

    def _load_german(self):
        german_data_path = os.path.join(CURR_DIR, "nmt_workload", "german_text.de")
        german_data_file = open(german_data_path, "rb")
        german_text = german_data_file.readlines()
        np.random.shuffle(german_text)
        return german_text

    def _load_reviews(self):
        base_path = os.path.join(CURR_DIR, "workload_data/aclImdb/test/")
        reviews = []
        pos_path = os.path.join(base_path, "pos")
        for rev_file in os.listdir(pos_path):
            with open(os.path.join(pos_path, rev_file), "r") as f:
                reviews.append(f.read().strip())

        neg_path = os.path.join(base_path, "neg")
        for rev_file in os.listdir(neg_path):
            with open(os.path.join(neg_path, rev_file), "r") as f:
                reviews.append(f.read().strip())
        # Shuffle in place
        np.random.shuffle(reviews)
        return reviews


if __name__ == "__main__":

    model = "nmt"
    num_clients = 1

    input_size = 40
    for cpus in [2, 3, 4, 5]:
        for gpus in [1]:
            for batch_size in [1, 2, 4, 6, 8, 10, 15, 20, 25]:
                model_config = get_heavy_node_config(model_name=model,
                                                    batch_size=batch_size,
                                                    num_replicas=1,
                                                    cpus_per_replica=cpus,
                                                    allocated_cpus=range(6, 16),
                                                    allocated_gpus=range(gpus),
                                                    input_size=input_size)
                setup_clipper(model_config)
                queue = Queue()
                benchmarker = ModelBenchmarker(model_config, queue)

                processes = []
                all_stats = []
                for _ in range(num_clients):
                    p = Process(target=benchmarker.run)
                    p.start()
                    processes.append(p)
                for p in processes:
                    all_stats.append(queue.get())
                    p.join()

                cl = ClipperConnection(DockerContainerManager(redis_port=6380))
                cl.connect()
                driver_utils.save_results([model_config],
                                        cl,
                                        all_stats,
                                        "single_model_prof_%s" % model_config.name)

    input_size = 80
    for cpus in [1, 2, 3, 4, 5]:
        for gpus in [1]:
            for batch_size in [1, 2, 4, 6, 8, 10, 15, 20, 25]:
                model_config = get_heavy_node_config(model_name=model,
                                                    batch_size=batch_size,
                                                    num_replicas=1,
                                                    cpus_per_replica=cpus,
                                                    allocated_cpus=range(6, 16),
                                                    allocated_gpus=range(gpus),
                                                    input_size=input_size)
                setup_clipper(model_config)
                queue = Queue()
                benchmarker = ModelBenchmarker(model_config, queue)

                processes = []
                all_stats = []
                for _ in range(num_clients):
                    p = Process(target=benchmarker.run)
                    p.start()
                    processes.append(p)
                for p in processes:
                    all_stats.append(queue.get())
                    p.join()

                cl = ClipperConnection(DockerContainerManager(redis_port=6380))
                cl.connect()
                driver_utils.save_results([model_config],
                                        cl,
                                        all_stats,
                                        "single_model_prof_%s" % model_config.name)
