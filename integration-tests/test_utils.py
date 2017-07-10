from __future__ import absolute_import, division, print_function
import os
import sys
import pprint
import random
import socket
import docker
import logging
import time
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath("%s/../clipper_admin_v2" % cur_dir))
import clipper_admin as cl
from clipper_admin import DockerContainerManager, K8sContainerManager
if sys.version < '3':
    import subprocess32 as subprocess
    PY3 = False
else:
    import subprocess
    PY3 = True


logger = logging.getLogger(__name__)

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
            logger.debug("randomly generated port %d is bound. Trying again." % port)



def init_clipper(container_manager, no_start=False):
    if container_manager == "docker":
        # TODO: create registry
        logging.info("Creating DockerContainerManager")
        cm = DockerContainerManager("localhost", redis_port=find_unbound_port())
        cl.stop_all(cm)
        docker_client = docker.from_env()
        docker_client.containers.prune(
            filters={"label": cl.container_manager.CLIPPER_DOCKER_LABEL})
        logging.info("Starting Clipper")
        # cl.start_clipper(cm)
        # time.sleep(1)
        # return cm
    elif container_manager == "k8s":
        logging.info("Creating K8sContainerManager")
        k8s_ip = subprocess.Popen(['minikube', 'ip'],
                                  stdout=subprocess.PIPE).communicate()[0].strip()
        logging.info("K8s IP: %s" % k8s_ip)
        cm = K8sContainerManager(k8s_ip, redis_port=find_unbound_port())
        cl.stop_all(cm)
    else:
        msg = "{cm} is a currently unsupported container manager".format(cm=container_manager)
        logging.error(msg)
        raise BenchmarkException(msg)
    if no_start:
        return cm
    logging.info("Starting Clipper")
    cl.start_clipper(cm)
    time.sleep(1)
    return cm


def log_clipper_state(cm):
    pp = pprint.PrettyPrinter(indent=4)
    logger.info("\nAPPLICATIONS:\n{app_str}".format(
        app_str=pp.pformat(cl.get_all_apps(cm, verbose=True))))
    logger.info("\nMODELS:\n{model_str}".format(
        model_str=pp.pformat(cl.get_all_models(cm, verbose=True))))
    logger.info("\nCONTAINERS:\n{cont_str}".format(
        cont_str=pp.pformat(cl.get_all_model_replicas(cm, verbose=True))))
