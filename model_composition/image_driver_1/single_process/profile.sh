#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 7 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 7 8 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 7 8 9 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 7 8 9 10 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 7 8 9 10 11 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)

(export CUDA_VISIBLE_DEVICES=0,1; numactl -C 1 2 3 4 5 6 7 8 9 10 12 python driver.py -c 1 -b 1 2 4 6 8 10 16 24 32 48 64 -r 0 -i 1 -t 8 -tl 100)