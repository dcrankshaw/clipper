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

from common import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-g', '--num_gpus', type=int, default=4, help="The number of GPUs available for use")
    parser.add_argument('-d', '--delay', type=float, default=None, help="A delay to begin with, without trying to find it with convergence")

    args = parser.parse_args()

    queue = Queue()

    ## FORMAT IS (INCEPTION, LOG REG, RESNET, KSVM)
    reps = [(1, 1, 1, 1),
            (1, 1, 2, 1),
            (1, 1, 3, 1),
            (1, 1, 4, 1),
            (2, 1, 4, 1),
            (2, 1, 5, 1),
            (2, 1, 6, 1)]


    ## FORMAT IS (INCEPTION, LOG REG, RESNET, KSVM)
    batches = (10, 1, 2, 16)

    latency_upper_bound = 1.0

    # ## THIS IS FOR 500MS
    # ## FORMAT IS (INCEPTION, LOG_REG, RESNET, KSVM)
    # five_hundred_ms_batches = (

    # five_hundred_ms_latency_upper_bound = 1.500

    # ## THIS IS FOR 375MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # three_seven_five_ms_reps = [(1, 1, 1, 1),
    #                             (1, 1, 1, 2),
    #                             (1, 1, 1, 3),
    #                             (1, 1, 1, 4),
    #                             (1, 1, 1, 5),
    #                             (2, 1, 1, 5),
    #                             (2, 1, 1, 6)]

    # ## THIS IS FOR 375MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # three_seven_five_ms_batches = (7, 2, 9, 2)

    # three_seven_five_ms_latency_upper_bound = 1.000

    # ## THIS IS FOR 375MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # thousand_ms_reps = [(1, 1, 1, 1),
    #                     (1, 1, 1, 2),
    #                     (2, 1, 1, 2),
    #                     (2, 1, 1, 3),
    #                     (2, 1, 1, 4),
    #                     (3, 1, 1, 4),
    #                     (3, 1, 1, 5)]

    # ## THIS IS FOR 1000MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # thousand_ms_batches = (16, 2, 16, 15)

    # thousand_ms_latency_upper_bound = 3.000

    inception_batch_idx = 0
    log_reg_batch_idx = 1
    resnet_batch_idx = 2
    ksvm_batch_idx = 3

    for inception_reps, log_reg_reps, resnet_reps, ksvm_reps in reps:
        total_cpus = range(9,29)

        def get_cpus(num_cpus):
            return [total_cpus.pop() for _ in range(num_cpus)]

        total_gpus = range(args.num_gpus)

        def get_gpus(num_gpus):
            return [total_gpus.pop() for _ in range(num_gpus)]

        resnet_cpus_per_replica = 2

        configs = [
            setup_inception(batch_size=batches[inception_batch_idx],
                            num_replicas=inception_reps,
                            cpus_per_replica=1,
                            allocated_cpus=get_cpus(inception_reps),
                            allocated_gpus=get_gpus(inception_reps)),
            setup_log_reg(batch_size=batches[log_reg_batch_idx],
                          num_replicas=log_reg_reps,
                          cpus_per_replica=1,
                          allocated_cpus=get_cpus(log_reg_reps),
                          allocated_gpus=[]),
            setup_kernel_svm(batch_size=batches[ksvm_batch_idx],
                             num_replicas=ksvm_reps,
                             cpus_per_replica=1,
                             allocated_cpus=get_cpus(ksvm_reps),
                             allocated_gpus=[]),
            setup_resnet(batch_size=batches[resnet_batch_idx],
                         num_replicas=resnet_reps,
                         cpus_per_replica=resnet_cpus_per_replica,
                         allocated_cpus=get_cpus(resnet_cpus_per_replica * resnet_reps),
                         allocated_gpus=get_gpus(resnet_reps))
        ]

        client_num = 0

        benchmarker = DriverBenchmarker(ID1_APP_NAME, configs, queue, latency_upper_bound, input_delay=args.delay)

        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        all_stats.append(queue.get())

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()

        fname = "incep_{}-logreg_{}-ksvm_{}-resnet_{}".format(inception_reps, log_reg_reps, ksvm_reps, resnet_reps)
        driver_utils.save_results(configs, cl, all_stats, "e2e_375ms_slo_img_driver_1", prefix=fname)

    sys.exit(0)
