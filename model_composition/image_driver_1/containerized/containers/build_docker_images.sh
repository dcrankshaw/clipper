#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

gcpprefix="gcr.io/clipper-model-comp"
tag="bench"

# Build RPC base images for python/anaconda and deep learning
# models
cd $DIR/../../../container_utils/
time docker build -t model-comp/py-rpc -f RpcDockerfile ./
time docker build -t model-comp/tf-rpc-uncompiled -f TfRpcDockerfile ./

tf_sha=$(nvidia-docker run -d model-comp/tf-rpc-uncompiled)

d=$(docker ps -q | wc -l)

while [ $d -ne "0" ]; do
        sleep 3
        d=$(docker ps -q | wc -l)
        docker ps
done

docker container commit $tf_sha model-comp/tf-rpc

time docker build -t model-comp/cuda-rpc -f CudaPyRpcDockerfile ./

cd $DIR
# Build model-specific images
time docker build -t model-comp/tf-kernel-svm -f TfKernelSvmDockerfile ./
docker tag model-comp/tf-kernel-svm $gcpprefix/tf-kernel-svm:$tag
# gcloud docker -- push $gcpprefix/tf-kernel-svm:$tag

time docker build -t model-comp/tf-resnet-feats -f TfResNetDockerfile ./
docker tag model-comp/tf-resnet-feats $gcpprefix/tf-resnet-feats:$tag
# gcloud docker -- push $gcpprefix/tf-resnet-feats:$tag

# time docker build -t model-comp/tf-resnet-feats-variable-input -f TfResNetVariableInputSizeDockerfile ./
# docker tag model-comp/tf-resnet-feats-variable-input $gcpprefix/tf-resnet-feats-variable-input:$tag
# # gcloud docker -- push $gcpprefix/tf-resnet-feats-variable-input:$tag
#
# time docker build -t model-comp/tf-resnet-feats-sleep -f TfResNetSleepDockerfile ./
# docker tag model-comp/tf-resnet-feats-sleep $gcpprefix/tf-resnet-feats-sleep:$tag
# # gcloud docker -- push $gcpprefix/tf-resnet-feats-sleep:$tag

time docker build -t model-comp/tf-log-reg -f TfLogisticRegressionDockerfile ./
docker tag model-comp/tf-log-reg $gcpprefix/tf-log-reg:$tag
# gcloud docker -- push $gcpprefix/tf-log-reg:$tag

time docker build -t model-comp/inception-feats -f InceptionFeaturizationDockerfile ./
docker tag model-comp/inception-feats $gcpprefix/inception-feats:$tag
# gcloud docker -- push $gcpprefix/inception-feats:$tag
