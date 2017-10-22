from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

# Inception feature vectors are of size 2048
INPUT_VECTOR_SIZE = 2048


def generate_bias():
    return (np.random.uniform(-1, 1) * 100).astype(dtype=np.float32)


def generate_weights():
    return np.random.uniform(-1, 1, size=(INPUT_VECTOR_SIZE, 1)).astype(dtype=np.float32)


def export():
    output_path = os.path.abspath("./exported_tf_models/log_reg_tfserve/1")
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            t_weights = tf.Variable(tf.convert_to_tensor(generate_weights()),
                                    name="weights", dtype=tf.float32)
            t_bias = tf.constant(-43.82, dtype=tf.float32)

            inputs_tensor = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])

            t_apply_weights = tf.reduce_sum(tf.multiply(
                t_weights, tf.transpose(inputs_tensor)), axis=0)
            t_sig_input = t_apply_weights + t_bias

            outputs_tensor = tf.sigmoid(t_sig_input)

        sess.run(tf.global_variables_initializer())

        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(inputs_tensor)
        outputs_tensor_info = tf.saved_model.utils.build_tensor_info(outputs_tensor)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': predict_inputs_tensor_info},
                outputs={
                    'outputs': outputs_tensor_info
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
