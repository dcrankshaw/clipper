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

tag=$(<VERSION.txt)
gcpprefix="gcr.io/clipper-model-comp"

# Build the Clipper Docker images
time docker build -t clipper/clipper-base:$tag -f ClipperBaseDockerfile ./
time docker build --build-arg CODE_VERSION=$tag -t clipper/zmq_frontend:$tag -f ZmqFrontendDockerfile ./
#docker tag clipper/zmq_frontend:$tag $gcpprefix/zmq_frontend:$tag
#gcloud docker -- push $gcpprefix/zmq_frontend:$tag
#exit
time docker build --build-arg CODE_VERSION=$tag -t clipper/management_frontend:$tag -f ManagementFrontendDockerfile ./
#docker tag clipper/management_frontend:$tag $gcpprefix/management_frontend:$tag
#gcloud docker -- push $gcpprefix/management_frontend:$tag
#cd -
