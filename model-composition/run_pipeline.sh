#!/usr/bin/env bash

redis-cli -p 6379 "flushall"
redis-cli -p 6380 "flushall"
python t1.py | xargs python t2.py | xargs python t3.py




