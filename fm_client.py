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

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from py.fm_ops import fm_ops


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  #user_id = "0000073794be9b4bd01a0d072834eaabdb0c5586"
  #doc_id_list = ["dfb4845_id"]
  vocab_size = 50000 
  #for doc_id in doc_id_list:
    # See prediction_service.proto for gRPC request/response details.
    #fea_str = user_id + " " + doc_id
  #fea_str = "0:1 1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 11:1 12:1 13:1 14:1 15:1 16:1 17:1 18:1 19:1 20:1 21:1 22:1 23:1 24:1 25:1 26:1 27:1 28:1 29:1 30:1 31:1 32:1 33:1 34:1 35:1 36:1"
  fea_str = "0 1;3 5 6" 
  fea = tf.placeholder(dtype = tf.string)
  ori_ids, feature_ids, feature_val, feature_pos = fm_ops.fm_line_parser(fea, vocab_size)
  with tf.Session() as sess:
    ori_ids_, feature_ids_, feature_val_, feature_pos_ = sess.run([ori_ids, feature_ids, feature_val, feature_pos], feed_dict = {fea: fea_str})

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'fm'
  request.model_spec.signature_name = 'predict_score'
  request.inputs['ori_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(ori_ids_))
  request.inputs['feature_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(feature_ids_))
  request.inputs['feature_vals'].CopyFrom(tf.contrib.util.make_tensor_proto(feature_val_))
  request.inputs['feature_pos'].CopyFrom(tf.contrib.util.make_tensor_proto(feature_pos_))

#    request.inputs['images'].CopyFrom(
#        tf.contrib.util.make_tensor_proto(data, shape=[1]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  print(result.outputs['pred_score'].float_val)


if __name__ == '__main__':
  tf.app.run()
