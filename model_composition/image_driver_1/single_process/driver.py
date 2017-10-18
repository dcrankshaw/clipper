import sys
import os
import argparse
import numpy as np
import json

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import driver_utils
from models import lgbm_model, vgg_feats_model, kpca_svm_model, inception_feats_model, kernel_svm_model

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

VGG_FEATS_MODEL_NAME = "vgg_feats"
INCEPTION_FEATS_MODEL_NAME = "inception_feats"
KERNEL_SVM_MODEL_NAME = "kernel_svm"
LGBM_MODEL_NAME = "lgbm"

VGG_MODEL_PATH = os.path.join(MODELS_DIR, "vgg_model_data", "vgg_feats_graph_def.pb")
KERNEL_SVM_MODEL_PATH = os.path.join(MODELS_DIR, "kernel_svm_model_data", "kernel_svm_trained.sav")
INCEPTION_MODEL_PATH = os.path.join(MODELS_DIR, "inception_model_data", "inception_feats_graph_def.pb")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model_data", "gbm_trained.sav")

TRIAL_LENGTH = 200

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, vgg_gpus=[], inception_gpus=[]):
    vgg_config = driver_utils.HeavyNodeConfig(model_name=VGG_FEATS_MODEL_NAME,
                                              input_type="floats",
                                              allocated_cpus=allocated_cpus,
                                              gpus=vgg_gpus,
                                              batch_size=batch_size)    

    inception_config = driver_utils.HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
                                                    input_type="floats",
                                                    allocated_cpus=allocated_cpus,
                                                    gpus=inception_gpus,
                                                    batch_size=batch_size)

    kernel_svm_config = driver_utils.HeavyNodeConfig(model_name=KERNEL_SVM_MODEL_NAME,
                                                     input_type="floats",
                                                     allocated_cpus=allocated_cpus,
                                                     gpus=[],
                                                     batch_size=batch_size)

    lgbm_config = driver_utils.HeavyNodeConfig(model_name=LGBM_MODEL_NAME,
                                               input_type="floats",
                                               allocated_cpus=allocated_cpus,
                                               gpus=[],
                                               batch_size=batch_size)

    return [vgg_config, inception_config, kernel_svm_config, lgbm_config]

def load_models(vgg_gpu, inception_gpu):
    models_dict = {
        VGG_FEATS_MODEL_NAME : create_vgg_model(VGG_MODEL_PATH, gpu_num=vgg_gpu),
        KERNEL_SVM_MODEL_NAME : create_svm_model(KERNEL_SVM_MODEL_PATH),
        INCEPTION_FEATS_MODEL_NAME : create_inception_model(INCEPTION_MODEL_PATH, gpu_num=inception_gpu),
        LGBM_MODEL_NAME : create_lgbm_model(LGBM_MODEL_PATH)
    }
    return models_dict

def create_vgg_model(model_path, gpu_num):
    return vgg_feats_model.VggFeaturizationModel(model_path, gpu_num=gpu_num)

def create_kernel_svm_model(model_path):
    return svm_model.KernelSVM(model_path)

def create_inception_model(model_path, gpu_num):
    return inception_feats_model.InceptionFeaturizationModel(model_path, gpu_num=gpu_num)

def create_lgbm_model(model_path):
    return lgbm_model.ImagesGBM(model_path)

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, models_dict):
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Stats
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []
        }
        self.total_num_complete = 0

        # Models
        self.vgg_model = models_dict[VGG_FEATS_MODEL_NAME]
        self.kernel_svm_model = models_dict[KERNEL_SVM_MODEL_NAME]
        self.inception_model = models_dict[INCEPTION_FEATS_MODEL_NAME]
        self.lgbm_model = models_dict[LGBM_MODEL_NAME]

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
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, vgg_inputs, inception_inputs):
        """
        Parameters
        ------------
        vgg_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 224 x 224 x 3
        inception_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 299 x 299 x 3
        """
        assert len(vgg_inputs) == len(inception_inputs)

        batch_size = len(vgg_inputs)

        begin_time = datetime.now()

        vgg_svm_future = self.thread_pool.submit(
            lambda inputs : self.kernel_svm_model.predict(self.vgg_model.predict(inputs)), vgg_inputs)
        
        inception_gbm_future = self.thread_pool.submit(
            lambda inputs : self.lgbm_model.predict(self.inception_model.predict(inputs)), inception_inputs)

        vgg_classes = vgg_future.result()
        inception_gbm_classes = inception_gbm_future.result()

        end_time = datetime.now()

        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += batch_size
        self.trial_num_complete += batch_size
        if self.trial_num_complete % TRIAL_LENGTH == 0:
            self.print_stats()
            self.init_stats()

class DriverBenchmarker(object):
    def __init__(self, models_dict):
        self.predictor = Predictor(models_dict)

    def set_configs(configs):
        self.configs = configs

    def run(self, num_trials, batch_size):
        logger.info("Generating random inputs")
        vgg_inputs = [self._get_vgg_feats_input() for _ in range(1000)]
        vgg_inputs = [i for _ in range(40) for i in vgg_inputs]

        inception_inputs = [self._get_inception_input() for _ in range(1000)]
        inception_inputs = [i for _ in range(40) for i in inception_inputs]

        assert len(inception_inputs) == len(vgg_inputs)
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.choice(len(vgg_inputs), batch_size)
            vgg_batch = vgg_inputs[batch_idx]
            inception_batch = vgg_inputs[batch_idx]

            self.predictor.predict(vgg_batch, inception_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        driver_utils.save_results(self.configs, [predictor.stats], "single_proc_gpu_and_batch_size_experiments")

    def _get_vgg_feats_input(self):
        # There's no need to flatten this input for a single-process model
        input_img = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return vgg_input

    def _get_inception_input(self):
        # There's no need to flatten this input for a single-process model
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-d', '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-v', '--vgg_gpu', type=int, default=0, help="The GPU on which to run the VGG featurization model")
    parser.add_argument('-i', '--inception_gpu', type=int, default=0, help="The GPU on which to run the inception featurization model")
    parser.add_argument('-t', '--num_trials', type=int, default=15, help="The number of trials to run")
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    
    models_dict = load_models(args.vgg_gpu, args.inception_gpu)
    benchmarker = DriverBenchmarker(models_dict)

    for batch_size in batch_size_confs:
        configs = get_heavy_node_configs(batch_size=batch_size,
                                         allocated_cpus=args.cpus,
                                         vgg_gpus=[args.vgg_gpu],
                                         inception_gpus=[args.inception_gpu])
        benchmarker.set_configs(configs)
        benchmarker.run(args.num_trials, batch_size)
