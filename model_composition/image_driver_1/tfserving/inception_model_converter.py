from __future__ import print_function
import os
import tensorflow as tf


def export():
    inception_model_path = os.path.abspath("./raw_data/inception_feats_graph_def.pb")
    output_path = os.path.abspath("./exported_tf_models/inception_tfserve/1")
    with tf.Session() as sess:
        with open(inception_model_path, mode='rb') as inception_file:
            inception_text = inception_file.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(inception_text)

        inputs_tensor = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        tf.import_graph_def(graph_def, name='', input_map={"ResizeBilinear:0": inputs_tensor})
        feats_tensor = tf.get_default_graph().get_tensor_by_name("pool_3:0")

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
