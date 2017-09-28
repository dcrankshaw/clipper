from __future__ import print_function
import sys
import os
import rpc
import base64

import numpy as np
import tensorflow as tf

class InceptionFeaturizationContainer(rpc.ModelContainerBase):

    def __init__(self, inception_model_path, gpu_mem_frac):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        graph = tf.get_default_graph()
        self.images_tensor = self._load_inception_model(inception_model_path)
        self.features_tensor = graph.get_tensor_by_name("pool_3:0")

    def predict_strings(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of byte-encoded jpeg images,
            represented a strings
        """

        # For a single input, inception returns a numpy array with dimensions
        # 1 x 1 x 1 x 2048, so we index into the containing arrays
        return [self._get_image_features(base64.b64decode(input_img))[0][0] for input_img in inputs]

    def _get_image_features(self, image):
        feed_dict = { self.images_tensor : image }
        features = self.sess.run(self.features_tensor, feed_dict=feed_dict)
        return features

    def _load_inception_model(self, inception_model_path):
        inception_file = open(inception_model_path, mode='rb')
        inception_text = inception_file.read()
        inception_file.close()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(inception_text)

        # Clear pre-existing device specifications
        # within the tensorflow graph
        for node in graph_def.node:
            node.device = ""

        with tf.device("/gpu:0"):
            # Create placeholder for an arbitrary number
            # of byte-encoded JPEG images
            images_tensor = tf.placeholder("string")
            tf.import_graph_def(graph_def, name='', input_map={ "DecodeJpeg/contents:0" : images_tensor})

        return images_tensor

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

    gpu_mem_frac = 1.0
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
    container = InceptionFeaturizationContainer(model_graph_path, gpu_mem_frac)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)

