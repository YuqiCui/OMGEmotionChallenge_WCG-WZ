#coding:utf-8

import tensorflow as tf
import numpy as np


def build_lstm(X_input=None, batch_size=32, images_num=20, num_layers=4, num_units=256*2,  dropout_rate=0.6, train=True, name='lstm', reuse=None,):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('LSTM_layer'):
            convnets = tf.reshape(X_input, shape=[-1, images_num, 2048], name='Reshape_for_lstm')
            # convnets = tf.nn.dropout(convnets, keep_prob=0.5, name='dropout_before_lstm')
            #lstm cell inputs:[batchs, time_steps, hidden_units]
            with tf.variable_scope('LSTM_Cell'):

                # def get_a_cell():
                #     return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True)

                def get_a_cell():
                    return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True), output_keep_prob=dropout_rate)

                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(num_layers)])       # batch_size32 无dropout，两层LSTM_256堆叠时，输出层为线性激活时，可达到ccc 0.247
                init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) #全初始化为0state
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, convnets, initial_state=init_state, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(outputs, [1,0,2])        # 调换0轴和1轴(time_steps, batch_size, num_units)
                outputs = outputs[-1]                           # 选取最后一个time_steps的输出outputs[-1]    shape[batch_size, num_units]
        
        h_fc2 = tf.layers.dense(outputs, num_units, activation=tf.nn.relu, name='fc_relu_256')
        h_fc3_drop = tf.nn.dropout(h_fc2, keep_prob=dropout_rate, name='dropout_3')
        # h_fc3 = tf.layers.dense(h_fc3_drop, 1, name='output')
        h_fc3 = tf.layers.dense(h_fc3_drop, 1, activation=tf.sigmoid, name='output')  
    
    return h_fc3


def build_lstm1(X_input=None, batch_size=32, images_num=20, num_layers=4, num_units=256*2,  dropout_rate=0.6, train=True, name='lstm', reuse=None,):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('LSTM_layer'):
            convnets = tf.reshape(X_input, shape=[-1, images_num, 2048], name='Reshape_for_lstm')
            #lstm cell inputs:[batchs, time_steps, hidden_units]
            with tf.variable_scope('LSTM_Cell'):

                def get_a_cell():
                    return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True), output_keep_prob=dropout_rate)

                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(num_layers)])
                init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, convnets, initial_state=init_state, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(outputs, [1,0,2])        # 调换0轴和1轴(time_steps, batch_size, num_units)
                outputs = outputs[-1]                           # 选取最后一个time_steps的输出outputs[-1]    shape[batch_size, num_units]


        # h_fc3 = tf.layers.dense(outputs, 1, name='output')
        h_fc3 = tf.layers.dense(outputs, 1, activation=tf.sigmoid, name='output')
    
    return h_fc3



# 单层LSTM网络
def build_lstm_single(X_input=None, batch_size=32, images_num=20, num_layers=1, num_units=256*2,  dropout_rate=0.6, train=True, name='lstm', reuse=None,):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('LSTM_layer'):
            convnets = tf.reshape(X_input, shape=[-1, images_num, 2048], name='Reshape_for_lstm')
            #lstm cell inputs:[batchs, time_steps, hidden_units]
            with tf.variable_scope('LSTM_Cell'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True)
                init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) #全初始化为0state
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, convnets, initial_state=init_state, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(outputs, [1,0,2])        # 调换0轴和1轴(time_steps, batch_size, num_units)
                outputs = outputs[-1]                           # 选取最后一个time_steps的输出outputs[-1]    shape[batch_size, num_units]

        h_drop = tf.nn.dropout(outputs, keep_prob=dropout_rate, name='dropout')
        # h_fc = tf.layers.dense(h_drop, 1, name='output')
        # h_fc = tf.layers.dense(h_drop, 1, activation=tf.sigmoid, name='output')
        h_fc = tf.layers.dense(h_drop, 1, activation=tf.nn.tanh, name='output')
    
    return h_fc


# 单层LSTM网络
def build_lstm_single_pca(X_input=None, batch_size=32, images_num=20, num_layers=1, num_units=256*2,  dropout_rate=0.6, train=True, name='lstm', reuse=None,):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('LSTM_layer'):
            convnets = tf.reshape(X_input, shape=[-1, images_num, 512], name='Reshape_for_lstm')
            #lstm cell inputs:[batchs, time_steps, hidden_units]
            with tf.variable_scope('LSTM_Cell'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True)
                init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) #全初始化为0state
                outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, convnets, initial_state=init_state, time_major=False)
                # outputs.shape is (batch_size, time_steps, num_units)
                outputs = tf.transpose(outputs, [1,0,2])        # 调换0轴和1轴(time_steps, batch_size, num_units)
                outputs = outputs[-1]                           # 选取最后一个time_steps的输出outputs[-1]    shape[batch_size, num_units]

        h_drop = tf.nn.dropout(outputs, keep_prob=dropout_rate, name='dropout')
        # h_fc = tf.layers.dense(h_drop, 1, name='output')
        # h_fc = tf.layers.dense(h_drop, 1, activation=tf.sigmoid, name='output')
        h_fc = tf.layers.dense(h_drop, 1, activation=tf.nn.tanh, name='output')
    
    return h_fc