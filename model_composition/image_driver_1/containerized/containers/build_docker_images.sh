#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build RPC base images for python/anaconda and deep learning
# models
cd $DIR/../../../container_utils/
time docker build -t model-comp/py-rpc -f RpcDockerfile ./
time docker build -t model-comp/tf-rpc -f TfRpcDockerfile ./
time docker build -t model-comp/tf-rpc:nogpu -f TfRpcNoGpuDockerfile ./

prefix="gcr.io/clipper-model-comp"
tag="bench"

cd $DIR
# Build model-specific images
time docker build -t $prefix/tf-kernel-svm:$tag -f TfKernelSvmDockerfile ./
gcloud docker -- push $prefix/tf-kernel-svm:$tag
time docker build -t $prefix/tf-resnet-feats:$tag -f TfResNetDockerfile ./
gcloud docker -- push $prefix/tf-resnet-feats:$tag
time docker build -t $prefix/tf-resnet-feats:$tag-nogpu -f TfResNetNoGpuDockerfile ./
gcloud docker -- push $prefix/tf-resnet-feats:$tag-nogpu
# time docker build -t model-comp/noop-sleep -f NoopDockerfile ./
# time docker build -t model-comp/noop-gpu -f NoopGPUDockerfile ./
time docker build -t $prefix/tf-log-reg:$tag -f TfLogisticRegressionDockerfile ./
gcloud docker -- push $prefix/tf-log-reg:$tag
# time docker build -t model-comp/vgg-feats -f VggFeaturizationDockerfile ./
# time docker build -t model-comp/kpca-svm -f VggKpcaSvmDockerfile ./
# time docker build -t model-comp/kernel-svm -f VggKernelSvmDockerfile ./
# time docker build -t model-comp/elastic-net -f VggElasticNetDockerfile ./
time docker build -t $prefix/inception-feats:$tag -f InceptionFeaturizationDockerfile ./
gcloud docker -- push $prefix/inception-feats:$tag
time docker build -t $prefix/inception-feats:$tag-nogpu -f InceptionFeaturizationNoGpuDockerfile ./
gcloud docker -- push $prefix/inception-feats:$tag-nogpu
time docker build -t $prefix/noop:$tag -f NoopDockerfile ./
gcloud docker -- push $prefix/noop:$tag
# time docker build -t model-comp/lgbm -f LgbmDockerfile ./
