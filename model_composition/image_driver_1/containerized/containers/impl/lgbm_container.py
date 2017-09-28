from __future__ import print_function
import sys
import os
import rpc

import lightgbm as lgb
import numpy as np

class ImagesGBMContainer(rpc.ModelContainerBase):

	def __init__(self, gbm_model_path):
		self.gbm_model = lgb.Booster(model_file=gbm_model_path)

	def predict_floats(self, inputs):
		"""
		Parameters
		----------
		inputs : list
			A list of inception feature vectors encoded
			as numpy arrays of type float32
		"""

		stacked_inputs = np.stack([input_item for input_item in inputs])
		all_classifications = self.gbm_model.predict(stacked_inputs)
		return [np.array(item, dtype=np.float32) for item in all_classifications]

if __name__ == "__main__":
	print("Starting LGBM Container")
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
		gbm_model_path = os.environ["CLIPPER_MODEL_PATH"]
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
	container = ImagesGBMContainer(gbm_model_path)
	rpc_service = rpc.RPCService()
	rpc_service.start(container, ip, port, model_name, model_version,
					  input_type)