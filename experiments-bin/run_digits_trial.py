from __future__ import print_function
import sys
import yaml
import toml
import time
import os
import json
# import shutil
import subprocess32 as subprocess


class DigitsBenchmarker:

    def __init__(self, experiment_name, log_dest, target_qps, batch_size, num_requests):
        self.cur_model_core_num = 0
        self.MAX_CORES = 47
        # self.NUM_REPS = NUM_REPS
        self.isolated_cores = True
        # self.experiment_name = "DEBUG_sklearn_logreg_%d_replicas_%s" % (NUM_REPS, str(time.strftime("%y%m%d-%H%M%S")))
        self.experiment_name = "%s_%s" % (experiment_name, str(time.strftime("%y%m%d-%H%M%S")))
        # self.benchmarking_logs = "benchmarking_logs/replica-scaling"
        self.benchmarking_logs = log_dest
        self.CLIPPER_ROOT = os.path.abspath("..")

        self.clipper_conf_dict = {
                "name" : self.experiment_name,
                "slo_micros" : 20000,
                "num_message_encodes" : 1,
                "correction_policy" : "logistic_regression",
                "use_lsh" : False,
                "input_type" : "float",
                "input_length" : 784,
                "window_size" : -1,
                "redis_ip" : "redis",
                "redis_port" : 6379,
                "results_path" : "/tmp/benchmarking_logs",
                "num_predict_workers" : 16,
                "num_update_workers" : 1,
                "cache_size" : 500000,
                "mnist_path" : "/mnist_data/test.data",
                # "num_benchmark_requests" : 10000000,
                "num_benchmark_requests" : num_requests,
                # "target_qps" : 10000*self.NUM_REPS,
                "target_qps" : target_qps,
                # "bench_batch_size" : 150*self.NUM_REPS,
                "bench_batch_size" : batch_size,
                "salt_cache" : True,
                "send_updates": False,
                "load_generator": "uniform",
                "request_generator": "balanced",
                # "cache_hit_rate": 1.0,
                # "batching": { "strategy": "aimd" },
                # "batching": { "strategy": "static", "batch_size": 1 },
                "batching": { "strategy": "learned", "sample_size": 500, "opt_addr": "quantilereg:7777"},
                "models": []
                }

        self.dc_dict = {
                "version": "2",
                "services": {
                    "redis": {"image": "redis:alpine", "cpuset": self.reserve_cores(1)},
                    "quantilereg": {"image": "clipper/quantile-reg", "cpuset": self.reserve_cores(1)},
                    "clipper": {"image": "cl-dev-digits",
                        "cpuset": self.reserve_cores(24),
                        "depends_on": ["redis", "quantilereg"],
                        "volumes": [
                            "${MNIST_PATH}:/mnist_data:ro",
                            "${CLIPPER_ROOT}/digits_bench.toml:/tmp/digits_bench.toml:ro",
                            "${CLIPPER_ROOT}/%s:/tmp/benchmarking_logs" % self.benchmarking_logs
                            ],
                        }
                    }
                }

    def reserve_cores(self, num_cores):
        # global cur_model_core_num
        s = "%d-%d" % (self.cur_model_core_num, self.cur_model_core_num + num_cores - 1)
        self.cur_model_core_num += num_cores
        if self.cur_model_core_num >= self.MAX_CORES:
            print("WARNING: Trying to reserve more than %d cores: %d" % (self.MAX_CORES, self.cur_model_core_num))
            sys.exit(1)
        return s

    def overlap_reserve_cores(self):
        s = "%d-%d" % (self.cur_model_core_num, self.MAX_CORES)
        return s

    def add_model(self, name_base, image, mp, container_mp, num_replicas=1):
        model_names = [name_base + "_r%d" % i for i in range(num_replicas)]
        model_addrs = ["%s:6001" % n for n in model_names]
        clipper_model_def = {
                "name": name_base,
                "addresses": model_addrs,
                "num_outputs": 1,
                "version": 1
        }
        dc_entries = {}
        for n in model_names:
            if self.isolated_cores:
                core_res = self.reserve_cores(1)
            else:
                core_res = self.overlap_reserve_cores()

            dc_entries[n] = {
                    "image": image,
                    "volumes": ["%s:/model:ro" % mp],
                    # "environment": ["CLIPPER_MODEL_PATH=%s" % container_mp],
                    "cpuset": core_res,
                    }

        # dc_entries should be added to the depends_on list

        # global clipper_conf_dict
        # global dc_dict
        self.clipper_conf_dict["models"].append(clipper_model_def)
        self.dc_dict["services"]["clipper"]["depends_on"].extend(model_names)
        self.dc_dict["services"].update(dc_entries)
    # return (clipper_model_def, model_names, dc_entries)

    def add_sklearn_rf(self, depth, num_replicas=1):
        name_base = "rf_d%d" % depth
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/50rf_pred3_depth%d/" % depth
        container_mp =  "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    def add_sklearn_log_regression(self, num_replicas=1):
        name_base = "logistic_reg"
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/log_regression_pred3/",
        container_mp =  "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    def add_sklearn_linear_svm(self, num_replicas=1):
        name_base = "linear_svm"
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/linearsvm_pred3/",
        container_mp =  "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    def add_sklearn_kernel_svm(self, num_replicas=1):
        name_base = "kernel_svm"
        image = "clipper/sklearn-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/svm_pred3/",
        container_mp = "/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    # def add_noop(self, num_replicas=1):
    #     name_base = "noop"
    #     image = "clipper/noop-mw"
    #     # these values don't matter
    #     model_path = "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/linearsvm_pred3/",
    #     container_mp = "/model"


    def add_spark_svm(self, num_replicas=1):
        name_base = "spark_svm"
        image = "clipper/spark-mw-dev"
        mp = "${CLIPPER_ROOT}/model_wrappers/python/spark_models/svm_predict_3",
        container_mp ="/model"
        self.add_model(name_base, image, mp, container_mp, num_replicas)

    def run_clipper(self):

        print("CORES USED: %d" % self.cur_model_core_num)

        # with open("../digits_bench_TEST.toml", 'w') as f:
        with open("../digits_bench.toml", 'w') as f:
            toml.dump(self.clipper_conf_dict, f)

        with open("docker-compose.yml", 'w') as f:
            yaml.dump(self.dc_dict, f)

        with open(os.path.join(self.CLIPPER_ROOT, self.benchmarking_logs, "%s_config.json" % self.experiment_name), "w") as f:
            json.dump({"clipper_conf": self.clipper_conf_dict, "docker_compose_conf": self.dc_dict}, f, indent=4)

        self.run_with_docker()

        print("Finished running. Saving Clipper log")

        with open(os.path.join(self.CLIPPER_ROOT, self.benchmarking_logs, "%s_logs.txt" % self.experiment_name), "w") as f:
            subprocess.call(["sudo", "docker", "logs", "experimentsbin_clipper_1"], stdout=f, stderr=subprocess.STDOUT, universal_newlines=True)


    def run_with_docker(self):
        """
        alias dcstop="sudo docker-compose stop"
        alias dcup="sudo docker-compose up clipper"
        """

        docker_compose_up = ["sudo", "docker-compose", "up", "clipper"]
        docker_compose_stop = ["sudo", "docker-compose", "stop"]
        print("Starting experiment: %s" % self.experiment_name)
        clipper_err_code = subprocess.call(docker_compose_up)
        if clipper_err_code > 0:
            print("WARNING: Clipper benchmark terminated with non-zero error code: %d" % clipper_err_code)
        print("Shutting down containers")
        dcstop_err_code = subprocess.call(docker_compose_stop)
        if dcstop_err_code > 0:
            print("WARNING: Docker container shutdown terminated with non-zero error code: %d" % dcstop_err_code)


# def gen_configs():
#
#     num_reps = 1
#     exp_name = "DEBUG_sklearn_logreg_%d_replicas" % num_reps
#     log_dest = "benchmarking_logs/replica-scaling"
#     # every 4 replicas add another million to the number of requests
#     # to let things stabilize
#     num_requests = 1000000 * (num_reps / 4 + 1)
#     benchmarker = DigitsBenchmarker(exp_name, log_dest, 10000*num_reps, 150*num_reps, num_requests)
#     benchmarker.add_sklearn_log_regression(num_replicas=num_reps)
#     benchmarker.run_clipper()
#
#     ## SKLEARN RF
#     # global cur_model_core_num
#     # num_reps = NUM_REPS
#     # add_spark_svm(num_replicas=num_reps)
#     # add_sklearn_rf(depth=16, num_replicas=num_reps)
#     # add_sklearn_linear_svm(num_replicas=num_reps)
#     # add_sklearn_rf(depth=8, num_replicas=num_reps)
#     # add_sklearn_kernel_svm(num_replicas=num_reps)
#
#
#     # cur_model_core_num = 0
#

if __name__=='__main__':
    # gen_configs()

    # num_reps = 1
    # for num_reps in [1,2] + range(4,37,4):
    for num_reps in range(16,33,4):
        print("STARTING RUN WITH %d REPLICAS" % num_reps)
        time.sleep(10)
        exp_name = "sklearn_logreg_%d_replicas" % num_reps
        log_dest = "benchmarking_logs/replica-scaling"
        # every 4 replicas add another million to the number of requests
        # to let things stabilize
        num_requests = 1000000 * (num_reps / 4 + 1)
        benchmarker = DigitsBenchmarker(exp_name, log_dest, 10000*num_reps, 150*num_reps, num_requests)
        benchmarker.add_sklearn_log_regression(num_replicas=num_reps)
        benchmarker.run_clipper()
        time.sleep(5)







# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
