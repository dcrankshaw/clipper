from __future__ import absolute_import, print_function
import os
import sys
import requests
import json
import numpy as np
import time
import logging
import docker

import findspark
findspark.init()
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession

from test_utils import (init_clipper, BenchmarkException,
                        headers, log_clipper_state)
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath("%s/../clipper_admin_v2" % cur_dir))
import clipper_admin as cl
from clipper_admin import DockerContainerManager
from clipper_admin.deployers.pyspark import deploy_pyspark_model

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

app_name = "pyspark_test"
model_name = "pyspark_model"

container_manager_type = "docker"


def normalize(x):
    return x.astype(np.double) / 255.0


def objective(y, pos_label):
    # prediction objective
    if y == pos_label:
        return 1
    else:
        return 0


def parseData(line, obj, pos_label):
    fields = line.strip().split(',')
    return LabeledPoint(
        obj(int(fields[0]), pos_label), normalize(np.array(fields[1:])))


def predict(spark, model, xs):
    return [str(model.predict(normalize(x))) for x in xs]


def deploy_and_test_model(sc, cm, model, version):
    deploy_pyspark_model(cm, model_name, version, "ints", predict, model, sc)
    time.sleep(25)
    num_preds = 25
    num_defaults = 0
    for i in range(num_preds):
        response = requests.post(
            "http://localhost:1337/%s/predict" % app_name,
            headers=headers,
            data=json.dumps({
                'input': get_test_point()
            }))
        result = response.json()
        if response.status_code == requests.codes.ok and result["default"]:
            num_defaults += 1
        elif response.status_code != requests.codes.ok:
            print(result)
            raise BenchmarkException(response.text)

    if num_defaults > 0:
        print("Error: %d/%d predictions were default" % (num_defaults,
                                                         num_preds))
    if num_defaults > num_preds / 2:
        raise BenchmarkException("Error querying APP %s, MODEL %s:%d" %
                                 (app_name, model_name, version))


def train_logistic_regression(trainRDD):
    return LogisticRegressionWithSGD.train(trainRDD, iterations=10)


def train_svm(trainRDD):
    return SVMWithSGD.train(trainRDD)


def train_random_forest(trainRDD, num_trees, max_depth):
    return RandomForest.trainClassifier(
        trainRDD, 2, {}, num_trees, maxDepth=max_depth)


def get_test_point():
    return [np.random.randint(255) for _ in range(784)]


if __name__ == "__main__":
    pos_label = 3
    try:
        spark = SparkSession\
                .builder\
                .appName("clipper-pyspark")\
                .getOrCreate()
        sc = spark.sparkContext
        cm = init_clipper(container_manager=container_manager_type)

        train_path = os.path.join(cur_dir, "data/train.data")
        trainRDD = sc.textFile(train_path).map(
            lambda line: parseData(line, objective, pos_label)).cache()

        try:
            cl.register_application(cm, app_name, model_name, "ints",
                                    "default_pred", 100000)
            time.sleep(1)
            response = requests.post(
                "http://localhost:1337/%s/predict" % app_name,
                headers=headers,
                data=json.dumps({
                    'input': get_test_point()
                }))
            result = response.json()
            if response.status_code != requests.codes.ok:
                print("Error: %s" % response.text)
                raise BenchmarkException("Error creating app %s" % app_name)

            version = 1
            lr_model = train_logistic_regression(trainRDD)
            deploy_and_test_model(sc, cm, lr_model, version)

            version += 1
            svm_model = train_svm(trainRDD)
            deploy_and_test_model(sc, cm, svm_model, version)

            version += 1
            rf_model = train_random_forest(trainRDD, 20, 16)
            deploy_and_test_model(sc, cm, svm_model, version)
        except BenchmarkException as e:
            log_clipper_state(cm)
            logger.exception("BenchmarkException")
            cl.stop_all(cm)
            docker_client = docker.from_env()
            docker_client.containers.prune(
                filters={"label": cl.container_manager.CLIPPER_DOCKER_LABEL})
            sys.exit(1)
        else:
            spark.stop()
            cl.stop_all(cm)
            docker_client = docker.from_env()
            docker_client.containers.prune(
                filters={"label": cl.container_manager.CLIPPER_DOCKER_LABEL})
            logger.info("ALL TESTS PASSED")
    except Exception as e:
        logger.exception("Exception")
        cm = DockerContainerManager("localhost")
        cl.stop_all(cm)
        docker_client = docker.from_env()
        docker_client.containers.prune(
            filters={"label": cl.container_manager.CLIPPER_DOCKER_LABEL})
        sys.exit(1)
