from __future__ import print_function
import sys
import os
import rpc
import base64

import numpy as np
import tensorflow as tf


class TensorFlowSavedModelContainer(rpc.ModelContainerBase):

    def __init__(self, model_path, gpu_mem_frac):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(graph=tf.Graph(),
                               config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        meta_graph_def = tf.saved_model.loader.load(self.sess,
                                                    [tf.saved_model.tag_constants.SERVING],
                                                    model_path)
        sigdef = tf.contrib.saved_model.get_signature_def_by_key(meta_graph_def, "serving_default")
        input_name = sigdef.inputs.get("inputs").name
        scores_name = sigdef.outputs.get("scores").name
        classes_name = sigdef.outputs.get("classes").name
        self.output_tensors = [self.sess.graph.get_tensor_by_name(scores_name),
                               self.sess.graph.get_tensor_by_name(classes_name)]

        self.input_tensor = self.sess.graph.get_tensor_by_name(input_name)

    def predict_strings(self, inputs):
        images = [base64.b64decode(input_img) for input_img in inputs]
        feed_dict = {self.input_tensor: images}
        results = self.sess.run(fetches=self.output_tensors, feed_dict=feed_dict)
        print(results)
        return ["done" for _ in range(len(inputs))]


if __name__ == "__main__":
    print("Starting Inception Featurization Container")
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
        model_graph_path = os.environ["CLIPPER_MODEL_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    gpu_mem_frac = .9
    if "CLIPPER_GPU_MEM_FRAC" in os.environ:
        gpu_mem_frac = float(os.environ["CLIPPER_GPU_MEM_FRAC"])
    else:
        print("Using all available GPU memory")

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

    input_type = "strings"
    container = TensorFlowSavedModelContainer(model_graph_path, gpu_mem_frac)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)
