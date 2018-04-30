#coding:utf-8

import os
import csv
import tensorflow as tf
import numpy as np
from read_data import load_feature_label
from read_data import load_feature_label_test
from calculateEvaluationCCC import ccc
import scipy.io

np.random.seed(1234)
tf.set_random_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset_path = '/mnt/disk2/ywang/omg_competition_images/feature/'

batch_size = 32*2
model_type = 'arousal'; checkpoint_path = '/home/wy/omg_final/runs/arousal/checkpoints/model'    # 0.33203
# model_type = 'valence'; checkpoint_path = '/home/wy/omg_final/runs/2018-04-29.10.49_arousal/checkpoints/model'


def iterate_minibatches(inputs, batchsize):
    input_len = inputs.shape[0]
    for start_idx in range(0, input_len, batchsize):
        if start_idx + batchsize >= input_len:
            excerpt = np.array(range(start_idx, input_len))
            excerpt = np.concatenate([excerpt,
                            np.tile([input_len-1], batchsize - input_len%batchsize)], axis=0)
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        print(excerpt)
        yield inputs[excerpt]

# load dataset
X_train, y_train, X_test, y_test = load_feature_label(dataset_path)
# X_test = load_feature_label_test(dataset_path)
print('-'*50)
if model_type == 'arousal':
    y_train = y_train[:,0].reshape(-1,1)    # arousal
    y_test = y_test[:,0].reshape(-1,1)      # arousal
elif model_type == 'valence':
    y_train = y_train[:,1].reshape(-1,1)    # valence
    y_test = y_test[:,1].reshape(-1,1)      # valence


# Evaluation
# ==================================================
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
        saver.restore(sess, checkpoint_path)

        # Get the placeholders from the graph by name
        input_var = graph.get_operation_by_name("Inputs/X_inputs").outputs[0]   # Tensor("Inputs/X_inputs:0", shape=(?, ?, 512), dtype=float32)

        tf_is_training = graph.get_operation_by_name("Inputs/is_training").outputs[0] 
        tf_keep_prob = graph.get_operation_by_name("Inputs/keep_prob").outputs[0]

        # Tensors we want to evaluate
        # predictions = graph.get_operation_by_name("lstm/output/BiasAdd").outputs[0] # Tensor("lstm/output/Sigmoid:0", shape=(64, 1), dtype=float32)
        predictions = graph.get_operation_by_name("lstm/output/Sigmoid").outputs[0] 
        # predictions = graph.get_operation_by_name("lstm/output/Tanh").outputs[0]
        print(predictions)

        # Collect the predictions here
        test_pre = []
        for batch in iterate_minibatches(X_test, batch_size):
            inputs = batch
            pred = sess.run(predictions, {input_var: inputs, tf_keep_prob: 1.0, tf_is_training: False})
            test_pre.extend(pred.reshape(-1))
            # print(inputs[:,0,0])
        print(pred.shape)
        print(np.array(test_pre).shape)
        

# Print accuracy if y_test is defined
if y_test is not None:
    ccc_test = ccc(y_test.reshape(-1), test_pre[:len(y_test)])
    print("Total number of test examples: {}".format(len(y_test)))
    print("CCC: {:g}".format(ccc_test))

# Save the evaluation to a csv

if model_type == 'arousal':
    out_path = "./prediction_arousal.csv"
    print("Saving evaluation to {0}".format(out_path))
    np.savetxt(out_path, test_pre, fmt='%.6f', delimiter=',')
    scipy.io.savemat('./omg_pred_arousal.mat', {'arousal': test_pre[:len(X_test)]})

elif model_type == 'valence':
    out_path = "./prediction_valence.csv"
    print("Saving evaluation to {0}".format(out_path))
    np.savetxt(out_path, test_pre, fmt='%.6f', delimiter=',')
    scipy.io.savemat('./omg_pred_valence.mat', {'valence': test_pre[:len(X_test)]})

