#!/usr/bin/env bash

python t1.py | xargs python t2.py | xargs python t3.py
