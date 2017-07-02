import os
import sys
import argparse

from transformer import *

redis_zero = SocketAddress("127.0.0.1", 6379)
redis_one = SocketAddress("127.0.0.1", 6380)

def run_source():
    print("Running source node")
    source = Source("source", SocketAddress("127.0.0.1", 7000), redis_zero, [redis_one])
    for line in sys.stdin:
        source.send(line.split(" "))

def double(tuples):
    return [Tuple(t.tuple_id, "%s%s" % (t.data, t.data)) for t in tuples]


def prepend_len(tuples):
    return [Tuple(t.tuple_id, "%d: %s" % (len(t.data), t.data)) for t in tuples]

def run_m1():
    print("Running m1")
    Transformer(
            "m1",
            double,
            SocketAddress("127.0.0.1", 7000),
            redis_zero,
            [redis_one]).run()

def run_m2():
    print("Running m2")
    Transformer(
            "m2",
            prepend_len,
            SocketAddress("127.0.0.1", 7000),
            redis_one,
            [redis_zero]).run()

def run_sink():
    print("Running sink")
    Sink(
        "sink",
        SocketAddress("127.0.0.1", 7000),
        redis_zero,
        [redis_one]).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example DAG")
    parser.add_argument('node')
    args = parser.parse_args()
    if args.node == "source":
        run_source()
    elif args.node == "m1":
        run_m1()
    elif args.node == "m2":
        run_m2()
    elif args.node == "sink":
        run_sink()
    else:
        print("%s is invalid node name. Must be one of source, m1, m2, sink" % args.node)

