#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

prefix="gcr.io/clipper-model-comp"
tag="bench"

# Build RPC base images for python/anaconda and deep learning
# models
cd $DIR/../../../container_utils/
time docker build -t model-comp/cuda-rpc -f CudaPyRpcDockerfile ./

cd $DIR
# Build model-specific images
time docker build -t $prefix/pytorch:$tag -f PyTorchDockerfile ./
# gcloud docker -- push $prefix/pytorch:$tag
time docker build -t $prefix/pytorch-alexnet:$tag -f AlexnetPyTorchDockerfile ./
# gcloud docker -- push $prefix/pytorch-alexnet:$tag
time docker build -t $prefix/pytorch-res50:$tag -f Res50PyTorchDockerfile ./
# gcloud docker -- push $prefix/pytorch-res50:$tag
time docker build -t $prefix/pytorch-res152:$tag -f Res152PyTorchDockerfile ./
# gcloud docker -- push $prefix/pytorch-res152:$tag

# time docker build -t $prefix/pytorch-no-rpc:$tag -f PyTorchNoRpcDockerfile ./
# gcloud docker -- push $prefix/pytorch-no-rpc:$tag
