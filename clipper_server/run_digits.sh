#!/usr/bin/env bash

./target/release/clipper digits --conf=features.toml --mnist=/crankshaw-local/mnist/data/test.data --users=500 --traindata=30 --testdata=100
