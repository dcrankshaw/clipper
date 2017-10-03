from __future__ import print_function
import sys
import os
import rpc
import pickle
import numpy as np

from elastic_net_model import ElasticNetModel

class ElasticNetContainer(rpc.ModelContainerBase):

	def __init__(self, en_model_path):
		en_model_file = open(ks_model_path, "rb")
		self.en_model = pickle.load(en_model_file)
		en_model_file.close()

	def predict_floats(self, inputs):
		"""
		Given a list of vgg feature vectors encoded as numpy arrays of data type
		np.float32, outputs a corresponding list of image category labels
		"""
		all_classifications = self.en_model.evaluate(inputs)
		return [np.array(item, dtype=np.float32) for item in all_classifications]

if __name__ == "__main__":
	print("Starting VGG Elastic Net Container")
	try:
		model_name = os.environ["CLIPPER_MODEL_NAME"]
	except KeyError:
		print(
			"ERROR: CLIPPER_MODEL_NAME environment variable must be set",
			file=sys.stdout)
		sys.exit(1)
	try:
		model_version = os.environ["CLIPPER_MODEL_VERSION"]
	except KeyError:
		print(
			"ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
			file=sys.stdout)
		sys.exit(1)
	try:
		en_model_path = os.environ["CLIPPER_MODEL_PATH"]
	except KeyError:
		print(
			"ERROR: CLIPPER_MODEL_PATH environment variable must be set",
			file=sys.stdout)
		sys.exit(1)

	ip = "127.0.0.1"
	if "CLIPPER_IP" in os.environ:
		ip = os.environ["CLIPPER_IP"]
	else:
		print("Connecting to Clipper on localhost")

	print("CLIPPER IP: {}".format(ip))

	port = 7000
	if "CLIPPER_PORT" in os.environ:
		port = int(os.environ["CLIPPER_PORT"])
	else:
		print("Connecting to Clipper with default port: 7000")

	input_type = "floats"
	container = ElasticNetContainer(en_model_path)
	rpc_service = rpc.RPCService()
	rpc_service.start(container, ip, port, model_name, model_version,
					  input_type)