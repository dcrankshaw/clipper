from __future__ import absolute_import, division, print_function
import os
import sys
import requests
import json
import numpy as np
import time
# import subprocess32 as subprocess
import pprint
import random
import socket
import docker
import logging
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath("%s/../clipper_admin_v2" % cur_dir))
# from clipper_admin import Clipper
import clipper_admin as cl
from clipper_admin import DockerContainerManager

headers = {'Content-type': 'application/json'}
fake_model_data = "/tmp/test123456"
try:
    os.mkdir(fake_model_data)
except OSError:
    pass


class BenchmarkException(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


# range of ports where available ports can be found
PORT_RANGE = [34256, 40000]


def find_unbound_port():
    """
    Returns an unbound port number on 127.0.0.1.
    """
    while True:
        port = random.randint(*PORT_RANGE)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except socket.error:
            print("randomly generated port %d is bound. Trying again." % port)


def init_clipper():
    # TODO: create registry

    logging.info("Creating DockerContainerManager")
    cm = DockerContainerManager("localhost", redis_port=find_unbound_port())
    # clipper = Clipper("localhost", redis_port=find_unbound_port())
    logging.info("Starting Clipper")
    cl.start_clipper(cm)
    time.sleep(1)
    return cm


def print_clipper_state(cm):
    pp = pprint.PrettyPrinter(indent=4)
    print("APPLICATIONS")
    pp.pprint(cl.get_all_apps(cm, verbose=True))
    print("\nMODELS")
    pp.pprint(cl.get_all_models(cm, verbose=True))
    print("\nCONTAINERS")
    pp.pprint(cl.get_all_containers(cm, verbose=True))


def deploy_model(cm, name, version):
    app_name = "%s_app" % name
    model_name = "%s_model" % name
    cl.deploy_model(
        cm,
        model_name,
        version,
        "doubles",
        fake_model_data,
        "clipper/noop-container",
        num_containers=1)
    time.sleep(10)
    num_preds = 25
    num_defaults = 0
    for i in range(num_preds):
        response = requests.post(
            "http://localhost:1337/%s/predict" % app_name,
            headers=headers,
            data=json.dumps({
                'input': list(np.random.random(30))
            }))
        result = response.json()
        if response.status_code == requests.codes.ok and result["default"]:
            num_defaults += 1
    if num_defaults > 0:
        print("Error: %d/%d predictions were default" % (num_defaults,
                                                         num_preds))
    if num_defaults > num_preds / 2:
        raise BenchmarkException("Error querying APP %s, MODEL %s:%d" %
                                 (app_name, model_name, version))


def create_and_test_app(cm, name, num_models):
    app_name = "%s_app" % name
    model_name = "%s_model" % name
    cl.register_application(cm, app_name, model_name, "doubles",
                            "default_pred", 100000)
    time.sleep(1)
    response = requests.post(
        "http://localhost:1337/%s/predict" % app_name,
        headers=headers,
        data=json.dumps({
            'input': list(np.random.random(30))
        }))
    result = response.json()
    if response.status_code != requests.codes.ok:
        print("Error: %s" % response.text)
        raise BenchmarkException("Error creating app %s" % app_name)

    for i in range(num_models):
        deploy_model(cm, name, i)
        time.sleep(1)


if __name__ == "__main__":
    num_apps = 6
    num_models = 8
    try:
        if len(sys.argv) > 1:
            num_apps = int(sys.argv[1])
        if len(sys.argv) > 2:
            num_models = int(sys.argv[2])
    except:
        # it's okay to pass here, just use the default values
        # for num_apps and num_models
        pass
    try:
        cm = init_clipper()
        try:
            print("Running integration test with %d apps and %d models" %
                  (num_apps, num_models))
            for a in range(num_apps):
                create_and_test_app(cm, "app_%s" % a, num_models)
            print(cl.get_clipper_logs(cm))
            print_clipper_state(cm)
            print("SUCCESS")
        except BenchmarkException as e:
            print_clipper_state(cm)
            print(e)
            cl.stop_all(cm)
            sys.exit(1)
        else:
            cl.stop_all(cm)
    except Exception as e:
        logging.error(e)
        cm = DockerContainerManager("localhost")
        cl.stop_all(cm)
        sys.exit(1)
