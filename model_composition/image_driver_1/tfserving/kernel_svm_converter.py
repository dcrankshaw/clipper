from __future__ import print_function
# import sys
import os
import numpy as np
import tensorflow as tf

# ResNet feature vectors are of size 2048
INPUT_VECTOR_SIZE = 2048
KERNEL_SIZE = 2000


def generate_bias():
    return (np.random.uniform(-1, 1) * 100).astype(dtype=np.float32)


def generate_weights():
    return np.random.uniform(-1, 1, size=(KERNEL_SIZE, 1)).astype(dtype=np.float32)


def generate_labels():
    return np.array(np.random.choice([-1, 1], size=(KERNEL_SIZE, 1)), dtype=np.float32)


def generate_kernel():
    return (np.random.rand(KERNEL_SIZE, INPUT_VECTOR_SIZE) * 10).astype(dtype=np.float32)


def export():
    output_path = os.path.abspath("./exported_tf_models/kernel_svm_tfserve/1")
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            t_kernel = tf.Variable(tf.convert_to_tensor(generate_kernel()), name="kernel", dtype=tf.float32)
            t_weights = tf.Variable(tf.convert_to_tensor(generate_weights()), name="weights", dtype=tf.float32)
            t_labels = tf.Variable(tf.convert_to_tensor(generate_labels()), name="labels", dtype=tf.float32)
            t_bias = tf.constant(-43.82, dtype=tf.float32)
            gamma = tf.constant(-50.0, dtype=tf.float32)

            inputs_tensor = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])

            rA = tf.reshape(tf.reduce_sum(tf.square(t_kernel), 1), [-1, 1])
            rB = tf.reshape(tf.reduce_sum(tf.square(inputs_tensor), 1), [-1, 1])
            pred_sq_dist = tf.add(tf.subtract(
                rA, tf.multiply(2.0, tf.matmul(t_kernel, tf.transpose(inputs_tensor)))),
                tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

            t_preds = tf.matmul(tf.multiply(tf.transpose(
                tf.multiply(t_labels, t_weights)), t_bias), pred_kernel)
            outputs_tensor = tf.sign(t_preds - tf.reduce_mean(t_preds))

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
