import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json

from clipper_admin import ClipperConnection, DockerContainerManager
from threading import Lock
from datetime import datetime
from io import BytesIO
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from containerized_utils.driver_utils import INCREASING, DECREASING, CONVERGED_HIGH, CONVERGED, UNKNOWN, CONVERGED_LOW
from multiprocessing import Process, Queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
INCEPTION_FEATS_MODEL_APP_NAME = "inception"
TF_KERNEL_SVM_MODEL_APP_NAME = "tf-kernel-svm"
TF_LOG_REG_MODEL_APP_NAME = "tf-log-reg"
TF_RESNET_MODEL_APP_NAME = "tf-resnet-feats"

INCEPTION_FEATS_IMAGE_NAME = "model-comp/inception-feats"
TF_KERNEL_SVM_IMAGE_NAME = "model-comp/tf-kernel-svm"
TF_LOG_REG_IMAGE_NAME = "model-comp/tf-log-reg"
TF_RESNET_IMAGE_NAME = "model-comp/tf-resnet-feats"

CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"

########## Setup ##########

def setup_clipper(configs):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="0,16,1,17,2,18,3,19")
    time.sleep(10)
    for config in configs:
        driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(20)
    logger.info("Clipper is set up!")
    return config

def setup_inception(batch_size,
                    num_replicas,
                    cpus_per_replica,
                    allocated_cpus,
                    allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=INCEPTION_FEATS_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=INCEPTION_FEATS_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )

def setup_log_reg(batch_size,
                  num_replicas,
                  cpus_per_replica,
                  allocated_cpus,
                  allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=TF_LOG_REG_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=TF_LOG_REG_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )

def setup_kernel_svm(batch_size,
                    num_replicas,
                    cpus_per_replica,
                    allocated_cpus,
                    allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=TF_KERNEL_SVM_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=TF_KERNEL_SVM_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )

def setup_resnet(batch_size,
                 num_replicas,
                 cpus_per_replica,
                 allocated_cpus,
                 allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=TF_RESNET_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=TF_RESNET_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )


########## Benchmarking ##########

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

    def __init__(self, clipper_metrics, batch_size):
        # The client through which the predictions are sent
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "all_lats": [],
        }
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
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        self.stats["all_lats"].append(self.latencies)
        if self.get_clipper_metrics:
            metrics = self.cl.inspect_instance()
            batch_sizes = get_batch_sizes(metrics)
            queue_sizes = get_queue_sizes(metrics)
            self.stats["mean_batch_sizes"].append(batch_sizes)
            self.stats["mean_queue_sizes"].append(queue_sizes)
            self.stats["all_metrics"].append(metrics)
            logger.info(("p99: {p99}, mean: {mean}, thruput: {thru}, "
                         "batch_sizes: {batches}").format(p99=p99, mean=mean, thru=thru,
                                                          batches=json.dumps(
                                                              batch_sizes, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))


    def ID1_predict(self, resnet_input, inception_input):
        begin_time = datetime.now()
        classifications_lock = Lock()
        classifications = {}

        def resnet_feats_continuation(resnet_features):
            if resnet_features == DEFAULT_OUTPUT:
                return
            return self.client.send_request(TF_KERNEL_SVM_MODEL_APP_NAME, resnet_features)

        def svm_continuation(svm_classification):
            if svm_classification == DEFAULT_OUTPUT:
                return
            else:
                classifications_lock.acquire()
                if TF_LOG_REG_MODEL_APP_NAME not in classifications:
                    classifications[TF_KERNEL_SVM_MODEL_APP_NAME] = svm_classification
                else:
                    update_perf_stats()
                classifications_lock.release()

        def inception_feats_continuation(inception_features):
            if inception_features == DEFAULT_OUTPUT:
                return
            return self.client.send_request(TF_LOG_REG_MODEL_APP_NAME, inception_features)


        def log_reg_continuation(log_reg_vals):
            if log_reg_vals == DEFAULT_OUTPUT:
                return
            else:
                classifications_lock.acquire()
                if TF_KERNEL_SVM_MODEL_APP_NAME not in classifications:
                    classifications[TF_LOG_REG_MODEL_APP_NAME] = log_reg_vals
                else:
                    update_perf_stats()
                classifications_lock.release()

        def update_perf_stats():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.trial_num_complete += 1

            trial_length = max(300, 10 * self.batch_size)
            if self.trial_num_complete % trial_length == 0:
                self.print_stats()
                self.init_stats()

        def try_catch(fun):
            def try_fun(x):
                try:
                    return fun(x)
                except Exception, e:
                    logger.error("Something bad happened!")
            return try_fun

        self.client.send_request(TF_RESNET_MODEL_APP_NAME, resnet_input) \
            .then(try_catch(resnet_feats_continuation)) \
            .then(try_catch(svm_continuation))

        self.client.send_request(INCEPTION_FEATS_MODEL_APP_NAME, inception_input) \
            .then(try_catch(inception_feats_continuation)) \
            .then(try_catch(log_reg_continuation))

class DriverBenchmarker(object):
    def __init__(self, configs, queue, latency_upper_bound, input_delay=None):
        # provided delay to being with. If None attempts to converge to find a value.
        self.input_delay = input_delay
        self.configs = configs
        self.max_batch_size = np.max([config.batch_size for config in configs])
        self.queue = queue
        logger.info("Generating random inputs")
        self.ID1_inputs = self.generate_ID1_inputs(1)
        self.latency_upper_bound = latency_upper_bound
        self.predictor = None
        self.iteration = 0
        self.total_iterations = 0

    def generate_ID1_inputs(self, size):
        def circular_repeat(base,num):
            base_size = base.shape[0]
            quotient = num // base_size
            remainder = num % base_size
            return np.concatenate([np.tile(base, (quotient,1)), base[:remainder]], axis=0)
        resnet_base = (np.random.rand(1000, 224, 224, 3) * 255).astype(np.float32).reshape(1000, -1)
        resnet_input = circular_repeat(resnet_base, size)
        inception_base = (np.random.rand(1000, 299, 299, 3) * 255).astype(np.float32).reshape(1000, -1)
        inception_input = circular_repeat(inception_base, size)
        return zip(resnet_input, inception_input)

    #### Mini-interface for using the Predictor in a decoupled way, just call these methods when needed ####
    def init_predictor(self):
        if self.predictor != None:
            raise Exception("Tried to intialize predictor twice")
        self.predictor = Predictor(clipper_metrics=True, batch_size=self.max_batch_size)

    def run_predictor(self):
        resnet_input, inception_input = self.ID1_inputs[self.iteration]
        self.predictor.ID1_predict(resnet_input, inception_input)
        self.iteration+=1
        self.total_iterations+=1
        if self.iteration % self.queries_per_sleep == 0:
                time.sleep(self.delay)
        self.iteration = self.iteration % len(self.ID1_inputs)
        if self.total_iterations%1000 == 0:
            logger.info("Sent for a total of {} predictions".format(self.total_iterations))

    def reset_predictor(self, wait = True):
        logger.info("Draining and resetting system...")
        self.cl.drain_queues()
        self.predictor.client.stop()
        del self.predictor
        self.predictor = None
        if wait:
            time.sleep(10)
    ########################################################################################################

    def warm_up_system(self):
        logger.info("Warming up system...")
        self.init_predictor()
        idx = 0
        # First warm up the model.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet, but we don't need as many warmup
        # iterations for the JIT-free models in this pipeline

        while len(self.predictor.stats["thrus"]) < 8:
            self.run_predictor()
        self.reset_predictor()

    # reduces the number of sleep calls, slightly batches results in two's though
    def scale_delay(self):
        if self.delay < 0.01:
            self.queries_per_sleep = 2
            self.delay = self.delay*2.0 - 0.001

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        self.init_predictor()
        while len(self.predictor.stats["thrus"]) < 10:
            self.run_predictor()
        # Now initialize request rate
        max_thruput = np.mean(self.predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        self.reset_predictor()

    def find_steady_state(self):

        self.init_predictor()
        time.sleep(10) # just to be extra sure I guess

        def increase_delay(self, multiple=1.0):
            if self.delay < 0.01:
                self.delay += 0.00005*multiple
            elif self.delay < 0.02:
                self.delay += 0.0002*multiple
            else:
                self.delay += 0.001*multiple

        idx = 0
        done = False
        # start checking for steady state after 10 trials
        last_checked_length = 10
        while not done:
            self.run_predictor()
            if len(self.predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(self.predictor.stats["thrus"]) + 1
                convergence_state = driver_utils.check_convergence_via_queue(
                    self.predictor.stats, self.configs, self.latency_upper_bound)
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    self.reset_predictor(wait=False)
                    return self.find_steady_state()
                elif convergence_state == CONVERGED:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.reset_predictor()
                    return
                elif len(self.predictor.stats) > 40:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    self.reset_predictor(wait=False)
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                elif convergence_state == CONVERGED_LOW:
                    logger.info("Converged LOW with delay of {}".format(self.delay))
                    logger.info("Consider re-running with smaller request delay augmentations")
                    raise Exception("LOW_CONVERGE, Consider re-running with smaller request delay augmentations")
                    done = True
                    self.reset_predictor(wait=False)
                    return
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)

    def run_more_predictions(self):
        self.init_predictor()
        for i in xrange(10000):
            if ((i+1) % 100) == 0:
                logger.info("Iteration "+str(i)+"...")
            self.run_predictor()
        time.sleep(10)
        return self.predictor.stats

    # new process
    def run(self):
        # initialize delay to be very small
        self.delay = 0.001
        self.queries_per_sleep = 1
        self.clipper_address = setup_clipper(self.configs)
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        # The observed amount of time needed to wait until the TF graphs are loaded on the GPU
        # This is time where progress isn't happening anyway, and we want to avoid making excessive prediction calls to clipper
        # and flood the frontend queue before the models are set up.
        time.sleep(230)
        logger.info("Waited 30 seconds to connect to clipper")
        self.warm_up_system()
        if self.input_delay != None:
            self.delay = self.input_delay
            self.scale_delay()
        else:
            logger.info("Initializing request rate (AKA delay)...")
            self.initialize_request_rate()
            self.scale_delay()
            logger.info("Initializing delay and queries_per_sleep to {delay} and {queries}".format(delay=self.delay,
                                                                           queries=self.queries_per_sleep))
            self.find_steady_state()
            logger.info("Found steady state...")
        logger.info("Final delay used: {delay}, queries_per_sleep: {queries}".format(delay=self.delay,
                                                                           queries=self.queries_per_sleep))
        final_stats = self.run_more_predictions()
        self.queue.put(final_stats)
        return
