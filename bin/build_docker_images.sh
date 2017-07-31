#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Let the user start this script from anywhere in the filesystem.
cd $DIR/..

tag="0.2-rc1"

# Build the Clipper Docker images
time docker build -t clipper/query_frontend:$tag -f QueryFrontendDockerfile ./
time docker build -t clipper/management_frontend:$tag -f ManagementFrontendDockerfile ./
cd -

# Build Spark JVM Container
cd $DIR/../containers/jvm
time docker build -t clipper/spark-scala-container:$tag -f SparkScalaContainerDockerfile ./
cd -

# Build the Python model containers
cd $DIR/..

# first build base image
docker build -t clipper/py-rpc:$tag -f ./RPCDockerfile ./
time docker build -t clipper/sum-container:$tag -f ./SumDockerfile ./
time docker build -t clipper/noop-container:$tag -f ./NoopDockerfile ./
time docker build -t clipper/python-closure-container:$tag -f ./PyClosureContainerDockerfile ./
time docker build -t clipper/pyspark-container:$tag -f ./PySparkContainerDockerfile ./
time docker build -t clipper/sklearn_cifar_container:$tag -f ./SklearnCifarDockerfile ./
time docker build -t clipper/tf_cifar_container:$tag -f ./TensorFlowCifarDockerfile ./
# TODO: uncomment
# time docker build -t clipper/r_python_container -f ./RPythonDockerfile ./
cd -
