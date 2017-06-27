#!/usr/bin/env bash


# rm -rf /tmp/redis
# mkdir -p /tmp/redis
# docker run -d -p 6379:6379 -v /tmp/redis:/run/redis clipper-mc/redis
docker run -d -p 6379:6379 redis:alpine
