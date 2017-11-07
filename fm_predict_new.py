from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from py.fm_ops import fm_ops
import os

model_path="/data01/share/wenjuand/fast_tffm/model"
model_version=3
export_path = os.path.join(
  tf.compat.as_bytes(model_path),
  tf.compat.as_bytes(str(model_version)))


sess = tf.Session()
meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
#model = sess.graph.get_tensor_by_name('vocab_block_0:0')
model = sess.graph.get_tensor_by_name('vocab_block_0:0')

#with sess.as_default():
  #model.eval().tofile('model.txt', sep='')
sess.run(tf.Print(model,[model],summarize=500))
