from __future__ import print_function
import sys
import os
import tensorflow as tf


def export():
    model_graph_path = os.path.abspath("./raw_data/tf_resnet_model_data/tf_resnet_152_feats_graph.meta")
    model_ckpt_path = os.path.abspath("./raw_data/tf_resnet_model_data/tf_resnet_152_feats.ckpt")
    output_path = os.path.abspath("./exported_tf_models/resnet_tfserve/1")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_graph_path, clear_devices=True)
        saver.restore(sess, model_ckpt_path)
        inputs_tensor = tf.get_default_graph().get_tensor_by_name('images:0')
        feats_tensor = tf.get_default_graph().get_tensor_by_name('avg_pool:0')

        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(inputs_tensor)
        feats_output_tensor_info = tf.saved_model.utils.build_tensor_info(feats_tensor)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': predict_inputs_tensor_info},
                outputs={
                    'feats': feats_output_tensor_info
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_inputs':
                    prediction_signature
            })

        builder.save()


if __name__ == "__main__":
    export()
