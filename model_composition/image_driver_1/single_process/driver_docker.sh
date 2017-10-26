#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

[ -d ${DIR}/results ] || mkdir ${DIR}/results

nvidia-docker run -d -v ${DIR}/results:/results model-comp/sp-img-driver-1 -t 12 -b 1 2 4 6 8 10 16 24 32 48 64 -c 17 -r 0 -i 1 -tl 100