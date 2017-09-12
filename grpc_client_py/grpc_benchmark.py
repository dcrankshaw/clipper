from __future__ import print_function

import sys
import grpc
import logging
import numpy as np
import time

import clipper_frontend_pb2
import clipper_frontend_pb2_grpc

from datetime import datetime
from multiprocessing import Process

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

CIFAR_SIZE_DOUBLES = 384

input_type = "doubles"
app_name = "app1"
model_name = "m1"

def run(proc_num):
	channel = grpc.insecure_channel('localhost:1337')
	stub = clipper_frontend_pb2_grpc.PredictStub(channel)
	i = 0
	latency = 0
	file_name = "/tmp/bench_{}".format(proc_num)
	while True:
		begin = datetime.now()
		x = clipper_frontend_pb2.DoubleData(data=list(np.random.random(CIFAR_SIZE_DOUBLES)))
		req = clipper_frontend_pb2.PredictRequest(application=app_name, data_type=DATA_TYPE_DOUBLES, double_data=x)
		response = stub.Predict(req)
		print("Received response!")
		end = datetime.now()

		latency += (end - begin).total_seconds()

		if i > 0 and i % 100 == 0:
			print("Throughput: {} qps\n".format(float(latency) / i))
			i = 0
			latency = 0

		i += 1

if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise

	num_procs = int(sys.argv[1])

	processes = []

	for i in range(num_procs):
		p = Process(target=run, args=('%d'.format(i),))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()
