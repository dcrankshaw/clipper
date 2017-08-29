from __future__ import print_function

import sys
import grpc
import logging
import numpy as np
import time

import clipper_frontend_pb2
import clipper_frontend_pb2_grpc

from datetime import datetime
from multiprocessing import Pool

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

CIFAR_SIZE = 384

input_type = "doubles"
app_name = "app1"
model_name = "m1"

def run():
	channel = grpc.insecure_channel('localhost:1337')
	stub = clipper_frontend_pb2_grpc.PredictStub(channel)
	i = 0
	while True:
		begin = datetime.now()
		x = clipper_frontend_pb2.DoubleData(data=list(np.random.random(CIFAR_SIZE)))
		req = clipper_frontend_pb2.PredictRequest(application=app_name, data_type=DATA_TYPE_DOUBLES, float_data=x)
		response = stub.Predict(req)
		end = datetime.now()

		latency += (end - begin).total_seconds()

	if i % 10000 == 0:
		print("Throughput: {} qps", float(latency) / i)
		i = 0
		latency = 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise

	num_procs = int(sys.argv[1])

	pool = Pool(num_procs)
	pool.map(run, [])
