from __future__ import print_function

import random
import time
import numpy as np

import grpc

import clipper_frontend_pb2
import clipper_frontend_pb2_grpc


# def guide_get_one_feature(stub, point):
#   feature = stub.GetFeature(point)
#   if not feature.location:
#     print("Server returned incomplete feature")
#     return
#
#   if feature.name:
#     print("Feature called %s at %s" % (feature.name, feature.location))
#   else:
#     print("Found no feature at %s" % feature.location)
#
#
# def guide_get_feature(stub):
#   guide_get_one_feature(stub, route_guide_pb2.Point(latitude=409146138, longitude=-746188906))
#   guide_get_one_feature(stub, route_guide_pb2.Point(latitude=0, longitude=0))


def run():
  channel = grpc.insecure_channel('localhost:1337')
  stub = clipper_frontend_pb2_grpc.PredictStub(channel)
  x = clipper_frontend_pb2.FloatsInput(input=list(np.random.random(100)))
  req = clipper_frontend_pb2.PredictRequest(application="m1", input=x)
  response = stub.PredictFloats(req)
  print(response)


if __name__ == '__main__':
  run()
