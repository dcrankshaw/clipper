from __future__ import absolute_import, division, print_function
import os
import sys
import requests
import json
import numpy as np
import time
import logging
from test_utils import (create_connection, BenchmarkException,
                        fake_model_data, headers, log_clipper_state, SERVICE)
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath("%s/../clipper_admin_v2" % cur_dir))

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)


def deploy_model(clipper_conn, name, version):
    app_name = "%s-app" % name
    model_name = "%s-model" % name
    clipper_conn.build_and_deploy_model(
        model_name,
        version,
        "doubles",
        fake_model_data,
        "clipper/noop-container",
        num_replicas=1,
        container_registry="568959175238.dkr.ecr.us-west-1.amazonaws.com/clipper")
    time.sleep(10)

    clipper_conn.link_model_to_app(app_name, model_name)
    time.sleep(30)

    num_preds = 25
    num_defaults = 0
    addr = clipper_conn.get_query_addr()
    for i in range(num_preds):
        response = requests.post(
            "http://%s/%s/predict" % (addr, app_name),
            headers=headers,
            data=json.dumps({
                'input': list(np.random.random(30))
            }))
        result = response.json()
        if response.status_code == requests.codes.ok and result["default"]:
            num_defaults += 1
    if num_defaults > 0:
        logger.error("Error: %d/%d predictions were default" % (num_defaults,
                                                                num_preds))
    if num_defaults > num_preds / 2:
        raise BenchmarkException("Error querying APP %s, MODEL %s:%d" %
                                 (app_name, model_name, version))


def create_and_test_app(clipper_conn, name, num_models):
    app_name = "%s-app" % name
    clipper_conn.register_application(app_name, "doubles", "default_pred", 100000)
    time.sleep(1)

    addr = clipper_conn.get_query_addr()
    response = requests.post(
        "http://%s/%s/predict" % (addr, app_name),
        headers=headers,
        data=json.dumps({
            'input': list(np.random.random(30))
        }))
    response.json()
    if response.status_code != requests.codes.ok:
        logger.error("Error: %s" % response.text)
        raise BenchmarkException("Error creating app %s" % app_name)

    for i in range(num_models):
        deploy_model(clipper_conn, name, i)
        time.sleep(1)


if __name__ == "__main__":
    num_apps = 6
    num_models = 8
    try:
        if len(sys.argv) > 1:
            num_apps = int(sys.argv[1])
        if len(sys.argv) > 2:
            num_models = int(sys.argv[2])
    except IndexError:
        # it's okay to pass here, just use the default values
        # for num_apps and num_models
        pass
    try:
        clipper_conn = create_connection(
            SERVICE, cleanup=True, start_clipper=True)
        time.sleep(10)
        print(clipper_conn.cm.get_query_addr())
        print(clipper_conn.inspect_instance())
        try:
            logger.info("Running integration test with %d apps and %d models" %
                        (num_apps, num_models))
            for a in range(num_apps):
                create_and_test_app(clipper_conn, "testapp%s" % a, num_models)
            logger.info(clipper_conn.get_clipper_logs())
            log_clipper_state(clipper_conn)
            logger.info("SUCCESS")
        except BenchmarkException as e:
            log_clipper_state(clipper_conn)
            logger.exception("BenchmarkException")
            create_connection(
                SERVICE, cleanup=True, start_clipper=False)
            sys.exit(1)
        else:
            create_connection(
                SERVICE, cleanup=True, start_clipper=False)
    except Exception as e:
        logger.exception("Exception")
        create_connection(SERVICE, cleanup=True, start_clipper=False)
        sys.exit(1)
