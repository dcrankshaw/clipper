# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

import numpy as np
from io import BytesIO
from PIL import Image

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
# tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
# FLAGS = tf.app.flags.FLAGS


def main():
    host, port = ("localhost", 9000)
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    # with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    # data = f.read()

    input_img = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    input_img = Image.fromarray(input_img.astype(np.uint8))
    inmem_inception_jpeg = BytesIO()
    resized_inception = input_img.resize((299, 299)).convert('RGB')
    resized_inception.save(inmem_inception_jpeg, format="JPEG")
    inmem_inception_jpeg.seek(0)
    data = inmem_inception_jpeg.read()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
    main()
