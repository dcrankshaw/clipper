import sys
import os
import argparse
import numpy as np
import json
import logging
import Queue
import time
import multiprocessing

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta
from threading import Thread, Lock

from single_proc_utils import HeavyNodeConfig, save_results
from models import tf_resnet_model, inception_feats_model, tf_kernel_svm_model, tf_log_reg_model

from e2e_utils import load_arrival_deltas, calculate_mean_throughput, calculate_peak_throughput

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

INCEPTION_FEATS_MODEL_NAME = "inception_feats"
TF_KERNEL_SVM_MODEL_NAME = "kernel_svm"
TF_LOG_REG_MODEL_NAME = "tf_log_reg"
TF_RESNET_MODEL_NAME = "tf_resnet_feats"

RESULTS_DIR = "/results"

INCEPTION_MODEL_PATH = os.path.join(MODELS_DIR, "inception_model_data", "inception_feats_graph_def.pb")
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "tf_resnet_model_data")

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, resnet_gpus=[], inception_gpus=[]):
    resnet_config = HeavyNodeConfig(model_name=TF_RESNET_MODEL_NAME,
                                    input_type="floats",
                                    allocated_cpus=allocated_cpus,
                                    gpus=resnet_gpus,
                                    batch_size=batch_size)

    inception_config = HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
                                       input_type="floats",
                                       allocated_cpus=allocated_cpus,
                                       gpus=inception_gpus,
                                       batch_size=batch_size)

    kernel_svm_config = HeavyNodeConfig(model_name=TF_KERNEL_SVM_MODEL_NAME,
                                        input_type="floats",
                                        allocated_cpus=allocated_cpus,
                                        gpus=[],
                                        batch_size=batch_size)

    log_reg_config = HeavyNodeConfig(model_name=TF_LOG_REG_MODEL_NAME,
                                     input_type="floats",
                                     allocated_cpus=allocated_cpus,
                                     gpus=[],
                                     batch_size=batch_size)

    return [resnet_config, inception_config, kernel_svm_config, log_reg_config]

def create_resnet_model(model_path, gpu_num):
    return tf_resnet_model.TfResNetModel(model_path, gpu_num)

def create_kernel_svm_model():
    return tf_kernel_svm_model.TFKernelSVM()

def create_inception_model(model_path, gpu_num):
    return inception_feats_model.InceptionFeaturizationModel(model_path, gpu_num=gpu_num)

def create_log_reg_model():
    return tf_log_reg_model.TfLogRegModel()

def load_models(resnet_gpu, inception_gpu):
    models_dict = {
        TF_RESNET_MODEL_NAME : create_resnet_model(RESNET_MODEL_PATH, gpu_num=resnet_gpu),
        TF_KERNEL_SVM_MODEL_NAME : create_kernel_svm_model(),
        INCEPTION_FEATS_MODEL_NAME : create_inception_model(INCEPTION_MODEL_PATH, gpu_num=inception_gpu),
        TF_LOG_REG_MODEL_NAME : create_log_reg_model()
    }
    return models_dict

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, models_dict):
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Stats
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "all_lats": [],
            "p99_predict_lats": [],
            "p99_queue_lats": [],
            "mean_batches": [],
            "mean_queue_lats": []
        }
        self.total_num_complete = 0

        # Models
        self.resnet_model = models_dict[TF_RESNET_MODEL_NAME]
        self.kernel_svm_model = models_dict[TF_KERNEL_SVM_MODEL_NAME]
        self.inception_model = models_dict[INCEPTION_FEATS_MODEL_NAME]
        self.log_reg_model = models_dict[TF_LOG_REG_MODEL_NAME]

    def init_stats(self):
        self.latencies = []
        self.predict_latencies = []
        self.queue_latencies = []
        self.batch_sizes = []
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        predict_lats = np.array(self.predict_latencies)
        queue_lats = np.array(self.queue_latencies)
        batch_sizes = np.array(self.batch_sizes)
        p99 = np.percentile(lats, 99)
        p99_predict = np.percentile(predict_lats, 99)
        p99_queue = np.percentile(queue_lats, 99)
        mean_queue = np.mean(queue_lats)
        mean_batch = np.mean(batch_sizes) 
        mean_lat = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["all_lats"].append(self.latencies)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean_lat)
        self.stats["p99_predict_lats"].append(self.predict_latencies)
        self.stats["p99_queue_lats"].append(p99_queue)
        self.stats["mean_queue_lats"].append(mean_queue)
        logger.info("p99: {p99}, p99_predict: {p99_pred}, p99_queue: {p99_queue}, "  
                "mean_queue: {mean_queue}, mean_lat: {mean_lat}, mean_batch: {mean_batch}, thruput: {thru}".format(
                                                                       p99=p99,
                                                                       mean_lat=mean_lat,
                                                                       mean_queue=mean_queue,
                                                                       mean_batch=mean_batch,
                                                                       thru=thru, 
                                                                       p99_pred=p99_predict,
                                                                       p99_queue=p99_queue))

    def predict(self, send_times, resnet_inputs, inception_inputs):
        """
        Parameters
        ------------
        send_times : [datetime]
            A list of timestamps at which each input was sent
        resnet_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 224 x 224 x 3
        inception_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 299 x 299 x 3
        """
        pred_begin = datetime.now()

        assert len(send_times) == len(resnet_inputs) == len(inception_inputs)

        for send_time in send_times:
            queue_lat = (pred_begin - send_time).total_seconds()
            self.queue_latencies.append(queue_lat)

        batch_size = len(resnet_inputs)
        self.batch_sizes.append(batch_size)

        resnet_svm_future = self.thread_pool.submit(
            lambda inputs : self.kernel_svm_model.predict(self.resnet_model.predict(inputs)), resnet_inputs)
        
        inception_log_reg_future = self.thread_pool.submit(
            lambda inputs : self.log_reg_model.predict(self.inception_model.predict(inputs)), inception_inputs)

        resnet_svm_classes = resnet_svm_future.result()
        inception_log_reg_classes = inception_log_reg_future.result()

        end_time = datetime.now()

        self.predict_latencies.append((end_time - pred_begin).total_seconds())

        for send_time in send_times:
            latency = (end_time - send_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1 
            self.trial_num_complete += 1

        TEMP_TRIAL_LENGTH = 100

        if self.trial_num_complete > TEMP_TRIAL_LENGTH:
            self.print_stats()
            self.init_stats()

class DriverBenchmarker(object):
    def __init__(self, trial_length, replica_configs, response_queue):
        self.trial_length = trial_length
        self.replica_configs = replica_configs
        self.response_queue = response_queue

    def run(self, num_trials, batch_size, process_file=None, request_delay=None):
        response_thread = Thread(target=self._run_async_response_service, args=())
        response_thread.start()
        if process_file: 
            self._benchmark_arrival_process(num_trials, process_file)
        elif request_delay:
            self._benchmark_over_under(num_trials, request_delay)
        else:
            raise

        response_thread.join()

    def _run_async_response_service(self):
        send_time = self.response_queue.get(block=True)
        recv_time = datetime.now()
        while self.response_queue.empty():
            try:
                send_time = self.response_queue.get_nowait()
                print((recv_time - send_time).total_seconds())
            except Queue.Empty:
                break

    def _get_load_balanced_replica_queue(self):
        replica_nums = self.replica_configs.keys()
        idx = np.random.randint(len(replica_nums))
        selected_num = replica_nums[idx]
        return self.replica_configs[selected_num][0]

    def _benchmark_arrival_process(self, num_trials, process_file):
        logger.info("Parsing arrival process")
        arrival_process = load_arrival_deltas(process_file)
        mean_throughput = calculate_mean_throughput(arrival_process)
        peak_throughput = calculate_peak_throughput(arrival_process)
        logger.info("Mean throughput: {}\nPeak throughput: {}".format(mean_throughput, peak_throughput))

        logger.info("Generating random inputs")
        resnet_inputs, inception_inputs = self._generate_inputs()
        
        logger.info("Starting predictions with specified arrival process")

        for idx in range(len(arrival_process)):
            for replica_num, queues in self.replica_configs.iteritems():
                feedback_queue = queues[1]
                if not feedback_queue.empty():
                    print(feedback_queue.get())
            
            input_idx = np.random.randint(len(inception_inputs))
            resnet_input = resnet_inputs[input_idx]
            inception_input = inception_inputs[input_idx]

            send_time = datetime.now()
            self._get_load_balanced_replica_queue().put((send_time, resnet_input, inception_input))

            request_delay = arrival_process[idx] * .001

            time.sleep(request_delay)

        processor_thread.join()

    def _benchmark_over_under(self, num_trials, request_delay):
        logger.info("Generating random inputs")
        resnet_inputs, inception_inputs = self._generate_inputs()
        
        logger.info("Starting predictions with a fixed request delay of: {} seconds".format(request_delay))

        for _ in range(len(resnet_inputs)):
            input_idx = np.random.randint(len(inception_inputs))
            resnet_input = resnet_inputs[input_idx]
            inception_input = inception_inputs[input_idx]

            send_time = datetime.now()
            self._get_load_balanced_replica_queue().put((send_time, resnet_input, inception_input))

            time.sleep(request_delay)
    
    def _generate_inputs(self):
        resnet_inputs = [self._get_resnet_feats_input() for _ in range(1000)]
        resnet_inputs = [i for _ in range(40) for i in resnet_inputs]

        inception_inputs = [self._get_inception_input() for _ in range(1000)]
        inception_inputs = [i for _ in range(40) for i in inception_inputs]

        assert len(inception_inputs) == len(resnet_inputs)
        return resnet_inputs, inception_inputs

    def _get_resnet_feats_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input.flatten()

    def _get_inception_input(self):
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input.flatten()

def run_async_query_processor(request_queue, response_queue, feedback_queue, models_dict, batch_size):
    try:
        predictor = Predictor(models_dict)
        queue_wait_times = []
        queue_sizes = []
        while True:
            curr_batch = []
            batch_item = request_queue.get(block=True)
            curr_batch.append(batch_item)
            while len(curr_batch) < batch_size and (not request_queue.empty()):
                try:
                    batch_item = request_queue.get_nowait()
                    curr_batch.append(batch_item)
                except Queue.Empty:
                    break


            send_times, resnet_inputs, inception_inputs = zip(*curr_batch)
            predictor.predict(send_times, resnet_inputs, inception_inputs)
            for send_time in send_times:
                response_queue.put(send_time)

    except Exception as e:
        feedback_queue.put(e)

########## Run Experiments ##########

def start_replica(replica_num, batch_size, request_queue, response_queue, feedback_queue):
    resnet_gpu, inception_gpu = get_gpus(replica_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{},{}".format(resnet_gpu, inception_gpu)
    models_dict = load_models(0, 1)
    run_async_query_processor(request_queue, response_queue, feedback_queue, models_dict, batch_size)

def get_gpus(replica_num):
    resnet_gpu = 2 * replica_num
    inception_gpu = (2 * replica_num) + 1
    return resnet_gpu, inception_gpu

def run_experiments(num_replicas, batch_size, num_trials, trial_length, process_file, request_delay):
    pool = multiprocessing.Pool(num_replicas)
    manager = multiprocessing.Manager()
    response_queue = manager.Queue()
   
    replica_configs = {}

    for replica_num in range(num_replicas):
        replica_request_queue = manager.Queue()
        replica_feedback_queue = manager.Queue()
        result = pool.apply_async(start_replica, (replica_num, batch_size, replica_request_queue, response_queue, replica_feedback_queue))
        replica_configs[replica_num] = (replica_request_queue, replica_feedback_queue)

    time.sleep(30)

    benchmarker = DriverBenchmarker(trial_length, replica_configs, response_queue)
    benchmarker.run(num_trials, batch_size, process_file, request_delay)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of VIRTUAL cpu cores on which to run the single process driver")
    parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-n',  '--num_replicas', type=int, help="The number of replicas to benchmark")
    parser.add_argument('-p',  '--process_file', type=str, help="Path to an arrival process file")
    parser.add_argument('-rd', '--request_delay', type=float, help="The request delay")
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs

    resnet_gpus = []
    inception_gpus = []
    for replica_num in range(args.num_replicas):
        resnet_gpu, inception_gpu = get_gpus(replica_num)
        resnet_gpus.append(resnet_gpu)
        inception_gpus.append(inception_gpu)

    for batch_size in batch_size_confs:
        configs = get_heavy_node_configs(batch_size=batch_size,
                                         allocated_cpus=args.cpus,
                                         resnet_gpus=resnet_gpus,
                                         inception_gpus=inception_gpus)
        run_experiments(args.num_replicas, batch_size, args.num_trials, args.trial_length, args.process_file, args.request_delay)
