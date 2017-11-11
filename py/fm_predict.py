import tensorflow as tf
import time, os
from py.fm_model import LocalFmModel, DistFmModel
from py.fm_ops import fm_ops

PREDICT_BATCH_SIZE = 1000

def predict(sess, predict_files, score_path, model_path, model_version, batch_size, vocabulary_size, hash_feature_id):
  with sess as sess:
      if not os.path.exists(score_path):
          os.mkdir(score_path)

      export_path = os.path.join(
          tf.compat.as_bytes(model_path),
          tf.compat.as_bytes(model_version))

      meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
      signature = meta_graph_def.signature_def
      signature_key = 'predict_score'
      ori_ids_tensor_name = signature[signature_key].inputs['ori_ids'].name
      feature_ids_tensor_name = signature[signature_key].inputs['feature_ids'].name
      feature_vals_tensor_name = signature[signature_key].inputs['feature_vals'].name
      feature_pos_tensor_name = signature[signature_key].inputs['feature_pos'].name
      pred_score_tensor_name = signature[signature_key].outputs['pred_score'].name


      ori_ids = sess.graph.get_tensor_by_name(ori_ids_tensor_name)
      feature_ids = sess.graph.get_tensor_by_name(feature_ids_tensor_name)
      feature_vals = sess.graph.get_tensor_by_name(feature_vals_tensor_name)
      feature_poses = sess.graph.get_tensor_by_name(feature_pos_tensor_name)
      pred_score = sess.graph.get_tensor_by_name(pred_score_tensor_name)

      file_id = tf.placeholder(dtype = tf.int32)
      data_file = tf.placeholder(dtype = tf.string)
      weight_file = tf.placeholder(dtype = tf.string)

      try:
          fid = 0
          for fname in predict_files:
              score_file = score_path + '/' + os.path.basename(fname) + '.score'
              print 'Start processing %s, scores written to %s ...'%(fname, score_file)
              labels_t, weights_t, ori_ids_t, feature_ids_t, feature_vals_t, feature_poses_t = fm_ops.fm_parser(file_id, data_file, weight_file, batch_size, vocabulary_size, hash_feature_id)
              with open(score_file, 'w') as o:
                   while True:
                        labels_, weights_, ori_ids_, feature_ids_, feature_vals_, feature_poses_ = sess.run([labels_t, weights_t, ori_ids_t, feature_ids_t, feature_vals_t, feature_poses_t], feed_dict = {file_id: fid, data_file: fname, weight_file: ''})
                        if len(labels_) == 0:
                            break
                        instance_score = sess.run(pred_score, {ori_ids: ori_ids_, feature_ids: feature_ids_, feature_vals: feature_vals_, feature_poses: feature_poses_})
                        for score in instance_score:
                            o.write(str(score) + '\n')
          fid += 1
      except tf.errors.OutOfRangeError:
          pass

def local_predict(predict_files, vocabulary_size, vocabulary_block_num, hash_feature_id, factor_num, model_file, score_path, model_path, model_version):
    predict(tf.Session(), predict_files, score_path, model_path, model_version, PREDICT_BATCH_SIZE, vocabulary_size, hash_feature_id)

