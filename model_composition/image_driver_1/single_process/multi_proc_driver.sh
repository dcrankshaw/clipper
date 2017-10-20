#!/usr/bin/env bash

set -e
set -u
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NUM_PROCS=$1

if [ "$NUM_PROCS" -ge "1" ]
	then
		echo '1'
		numactl -C 1,2,3,4 python driver.py -c 1 2 3 4 -v 0 -i 1 -b 32 &
fi

if [ "$NUM_PROCS" -ge "2" ]
	then
		echo '2'
		numactl -C 5,6,7,8 python driver.py -c 5 6 7 8 -v 2 -i 3 -b 32 &
fi

if [ "$NUM_PROCS" -ge "3" ]
	then
		echo '3'
		numactl -C 9,10,11,12 python driver.py -c 9 10 11 12 -v 4 -i 5 -b 32 &
fi

if [ "$NUM_PROCS" -ge "4" ]
	then
		echo '4'
		numactl -C 13,14,15,16 python driver.py -c 13 14 15 16 -v 6 -i 7 -b 32 &
fi