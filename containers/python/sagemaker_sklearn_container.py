from __future__ import print_function
import rpc
import os
import sys
import numpy as np
import cloudpickle
import boto3
import botocore
import tarfile

IMPORT_ERROR_RETURN_CODE = 3


def load_from_s3(file_path):
    # If these environment variables are in scope, will boto find them?
    aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    print(aws_access_key)
    print(aws_secret_key)

    # Strip s3://
    stripped_path = file_path.split("s3://")[1]

    components = stripped_path.split("/")
    bucket = components[0]
    key = "/".join(components[1:])

    s3 = boto3.client("s3")
    download_dir = "/model"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = os.path.join(download_dir, components[-1])
    with open(download_path, "wb") as f:
        s3.download_fileobj(bucket, key, f)

    if tarfile.is_tarfile(download_path):
        with tarfile.open(download_path) as tar:
            extract_dir = os.path.join(download_dir)
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            tar.extractall(path=extract_dir)

    model_path = None
    for fname in os.listdir(extract_dir):
        ext = os.path.splitext(fname)
        if ext[1] == ".pkl":
            model_path = os.path.join(extract_dir, fname)
            break

    return model_path


def load_sklearn_model_func(file_path):
    if file_path.startswith("s3://"):
        file_path = load_from_s3(file_path)
    print(file_path)

    if sys.version_info < (3, 0):
        with open(file_path, 'r') as model_file:
            return cloudpickle.load(model_file)
    else:
        with open(file_path, 'rb') as model_file:
            return cloudpickle.load(model_file)


class PythonContainer(rpc.ModelContainerBase):
    def __init__(self, path, input_type):
        self.input_type = rpc.string_to_input_type(input_type)
        # modules_folder_path = "{dir}/modules/".format(dir=path)
        # sys.path.append(os.path.abspath(modules_folder_path))
        # predict_fname = "func.pkl"
        # predict_path = "{dir}/{predict_fname}".format(
        #     dir=path, predict_fname=predict_fname)
        self.model = load_sklearn_model_func(path)

    def predict_ints(self, inputs):
        preds = self.model.predict(inputs)
        return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = self.model.predict(inputs)
        return [str(p) for p in preds]

    def predict_doubles(self, inputs):
        preds = self.model.predict(inputs)
        return [str(p) for p in preds]

    def predict_bytes(self, inputs):
        preds = self.model.predict(inputs)
        return [str(p) for p in preds]

    def predict_strings(self, inputs):
        preds = self.model.predict(inputs)
        return [str(p) for p in preds]


if __name__ == "__main__":
    print("Starting Python Closure container")
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

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: {port}".format(
            port=port))

    input_type = "doubles"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]
    else:
        print("Using default input type: doubles")

    model_path = os.environ["CLIPPER_MODEL_PATH"]

    print("Initializing Python function container")
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        # input_type = "floats"
        # model_path = "s3://sagemaker-us-east-1-568959175238/output/decision-trees-sample-2018-05-25-00-30-39-674/output/model.tar.gz"
        model = PythonContainer(model_path, input_type)
        rpc_service = rpc.RPCService()
        rpc_service.start(model, ip, port, model_name, model_version,
                          input_type)
    except ImportError as e:
        print(e)
        sys.exit(IMPORT_ERROR_RETURN_CODE)
