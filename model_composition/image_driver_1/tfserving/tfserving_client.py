# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

import os
import json
import numpy as np
# from io import BytesIO
# from PIL import Image
import logging
from datetime import datetime
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from multiprocessing import Process, Queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)


def gen_imagenet_image():

    input_img = np.array(np.random.rand(2048), dtype=np.float32)
    # input_img = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
    return input_img


def run_benchmark(q):
    bench = Benchmarker(q)
    bench.run()


class Benchmarker(object):
    def __init__(self, queue):
        host = "localhost"
        port = 9000
        self.channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.images = [gen_imagenet_image() for _ in range(100)]
        self.queue = queue
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []}

    def init_stats(self):
        self.latencies = []
        self.batch_num_complete = 0
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

    def run(self, num_trials=7, reqs_per_trial=100):
        self.init_stats()
        for t in range(num_trials):
            for r in range(reqs_per_trial):
                data = self.images[r % len(self.images)]
                request = predict_pb2.PredictRequest()
                request.model_spec.name = 'log_reg'
                request.model_spec.signature_name = 'predict_inputs'
                # TODO: remove this copy
                request.inputs['inputs'].CopyFrom(
                    tf.contrib.util.make_tensor_proto(data, shape=[1, 2048]))
                rstart = datetime.now()
                result = self.stub.Predict(request, 5.0)
                print(result)
                return
                lat = (datetime.now() - rstart).total_seconds()
                self.latencies.append(lat)
                self.batch_num_complete += 1
            self.print_stats()
            self.init_stats()
            # logger.info("Completed {} of {} trials".format(t, num_trials))
        self.queue.put(self.stats)


def mean_throughput(stats):
    mean_thru_per_client = [np.mean(s["thrus"][1:]) for s in stats]
    var_thru_per_client = [np.var(s["thrus"][1:]) for s in stats]
    mean_total = np.sum(mean_thru_per_client)
    std_total = np.sqrt(np.sum(var_thru_per_client))
    return (mean_total, std_total)


if __name__ == '__main__':

    # for num_procs in [1, 5, 10, 15, 20]:
    num_procs = 1
    q = Queue()
    processes = []
    stats = []
    for _ in range(num_procs):
        p = Process(target=run_benchmark, args=(q,))
        p.start()
        processes.append(p)
    for p in processes:
        stats.append(q.get())
        p.join()

    results_dir = os.path.abspath("results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = os.path.join(results_dir,
                                "{procs}-clients_{time:%y%m%d_%H%M%S}.json".format(
                                    procs=num_procs, time=datetime.now()))

    with open(results_file, "w") as f:
        json.dump(stats, f, indent=4)
        logger.info("Saved results to {}".format(results_file))
        mean_total, std_total = mean_throughput(stats)
    logger.info("Num clients: {}, mean throughput: {} +- {}".format(
        num_procs, mean_total, std_total))
