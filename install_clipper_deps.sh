#!/usr/bin/env bash

apt-get update && apt-get install -y g++ automake autoconf autoconf-archive libtool libboost-all-dev \
        libevent-dev libdouble-conversion-dev libgoogle-glog-dev libgflags-dev liblz4-dev \
        liblzma-dev libsnappy-dev make zlib1g-dev binutils-dev libjemalloc-dev libssl-dev \
        pkg-config libiberty-dev git cmake libev-dev libhiredis-dev libzmq5 libzmq5-dev
