from __future__ import print_function
import rpc
import os
import sys
import numpy as np
import time


class NoopContainer(rpc.ModelContainerBase):
    def __init__(self, prediction="1.0"):
        self.prediction = prediction

    def _predict(self, inputs):
        return [np.array(np.random.rand(2048), dtype=np.float32) for i in inputs]
        # return [self.prediction] * len(inputs)

    def predict_ints(self, inputs):
        return self._predict(inputs)

    def predict_floats(self, inputs):
        return self._predict(inputs)

    def predict_doubles(self, inputs):
        return self._predict(inputs)

    def predict_bytes(self, inputs):
        return self._predict(inputs)

    def predict_strings(self, inputs):
        return self._predict(inputs)


if __name__ == "__main__":
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

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    input_type = "floats"
    model = NoopContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, model_name, model_version, input_type)
