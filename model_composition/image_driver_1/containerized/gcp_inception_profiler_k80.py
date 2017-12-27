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
from containerized_utils.driver_utils import INCREASING, DECREASING, CONVERGED_HIGH, CONVERGED, UNKNOWN
from multiprocessing import Process, Queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

INCEPTION_FEATS_MODEL_APP_NAME = "inception-k80"
TF_KERNEL_SVM_MODEL_APP_NAME = "tf-kernel-svm"
TF_LOG_REG_MODEL_APP_NAME = "tf-log-reg"
TF_RESNET_MODEL_APP_NAME = "tf-resnet-feats"

CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"

GCP_CLUSTER_NAME = "single-model-profiles-inception-k80"


"""
Models:
    + Driver 1:
        + inception
        + resnet 152
        + logreg
        + kernel svm

"""

def setup_inception(batch_size,
                    num_replicas,
                    cpus_per_replica,
                    gpu_type):

    return driver_utils.HeavyNodeConfigGCP(name=INCEPTION_FEATS_MODEL_APP_NAME,
                                           input_type="floats",
                                           model_image="gcr.io/clipper-model-comp/inception-feats:bench",
                                           cpus_per_replica=cpus_per_replica,
                                           gpu_type=gpu_type,
                                           batch_size=batch_size,
                                           num_replicas=num_replicas,
                                           no_diverge=True)

def setup_log_reg(batch_size,
                  num_replicas,
                  cpus_per_replica):

    return driver_utils.HeavyNodeConfigGCP(name=TF_LOG_REG_MODEL_APP_NAME,
                                           input_type="floats",
                                           model_image="gcr.io/clipper-model-comp/tf-log-reg:bench",
                                           cpus_per_replica=cpus_per_replica,
                                           gpu_type=None,
                                           batch_size=batch_size,
                                           num_replicas=num_replicas,
                                           no_diverge=True)

def setup_kernel_svm(batch_size,
                    num_replicas,
                    cpus_per_replica):

    return driver_utils.HeavyNodeConfigGCP(name=TF_KERNEL_SVM_MODEL_APP_NAME,
                                           input_type="floats",
                                           model_image="gcr.io/clipper-model-comp/tf-kernel-svm:bench",
                                           cpus_per_replica=cpus_per_replica,
                                           gpu_type=None,
                                           batch_size=batch_size,
                                           num_replicas=num_replicas,
                                           no_diverge=True)

def setup_resnet(batch_size,
                 num_replicas,
                 cpus_per_replica,
                 gpu_type):

    return driver_utils.HeavyNodeConfigGCP(name=TF_RESNET_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image="gcr.io/clipper-model-comp/tf-resnet-feats:bench",
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
    clipper_address = cl.cm.query_frontend_external_ip
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

        def on_complete(output):
            if output == DEFAULT_OUTPUT:
                return
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1

            trial_length = max(300, 10 * self.batch_size)
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
        self.input_generator_fn = self._get_input_generator_fn(model_app_name=self.config.name)
        base_inputs = [self.input_generator_fn() for _ in range(1000)]
        self.inputs = base_inputs
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
        self.clipper_address = setup_clipper_gcp(self.config)
        self.cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        self.cl.connect()
        time.sleep(30)
        predictor = Predictor(self.clipper_address, clipper_metrics=True, batch_size=self.max_batch_size)
        idx = 0
        while len(predictor.stats["thrus"]) < 8:
            predictor.predict(self.config.name, self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        logger.info("Initializing delay to {}".format(self.delay))
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor

    def increase_delay(self):
        if self.delay < 0.005:
            self.delay += 0.0005
        else:
            self.delay += 0.001

    def find_steady_state(self):
        self.cl.cm.reset()
        time.sleep(30)
        logger.info("Clipper is reset")
        predictor = Predictor(self.clipper_address, clipper_metrics=True, batch_size=self.max_batch_size)
        idx = 0
        done = False
        # start checking for steady state after 7 trials
        last_checked_length = 12
        while not done:
            predictor.predict(self.config.name, self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 4
                convergence_state = driver_utils.check_convergence(predictor.stats, [self.config,], self.latency_upper_bound)
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    return self.find_steady_state()
                elif convergence_state == CONVERGED:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.queue.put((self.clipper_address, predictor.stats))
                    return
                elif len(predictor.stats) > 60:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)


    def _get_inception_input(self):
        input_img = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return input_img.flatten()

    def _get_tf_kernel_svm_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_log_reg_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_resnet_input(self):
        input_img = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return input_img.flatten()

    def _get_input_generator_fn(self, model_app_name):
        if model_app_name == INCEPTION_FEATS_MODEL_APP_NAME:
            return self._get_inception_input
        elif model_app_name == TF_KERNEL_SVM_MODEL_APP_NAME:
            return self._get_tf_kernel_svm_input
        elif model_app_name == TF_LOG_REG_MODEL_APP_NAME:
            return self._get_tf_log_reg_input
        elif model_app_name == TF_RESNET_MODEL_APP_NAME:
            return self._get_tf_resnet_input

if __name__ == "__main__":

    queue = Queue()

    # for gpu_type in ["k80", "p100"]:
    gpu_type = "k80"
        # for num_cpus in [1, 2]:
    num_cpus = 2
    for batch_size in [1, 2, 4, 8, 12, 16, 20, 24, 32]:
        config = setup_inception(batch_size, 1, num_cpus, gpu_type)
        client_num = 0
        benchmarker = DriverBenchmarker(config, queue, client_num, 0.1*batch_size)

        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        clipper_address, stats = queue.get()
        all_stats.append(stats)

        cl = ClipperConnection(GCPContainerManager(GCP_CLUSTER_NAME))
        cl.connect()

        fname = "results-{gpu}-{num_cpus}-{batch}".format(gpu=gpu_type, num_cpus=num_cpus, batch=batch_size)
        driver_utils.save_results([config,], cl, all_stats, "inception_smp_gcp", prefix=fname)

    sys.exit(0)
