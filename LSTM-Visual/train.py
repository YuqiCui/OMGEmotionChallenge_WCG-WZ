#coding:utf-8
import os
import time
import tensorflow as tf
import numpy as np
import scipy.io
import datetime
import matplotlib.pyplot as plt

from calculateEvaluationCCC import ccc
from read_data import load_feature_label
from read_data import iterate_minibatches

from model import build_lstm
from model import build_lstm1
from model import build_lstm_single


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1234)
tf.set_random_seed(1)

file_path = '/mnt/disk2/ywang/omg_competition_images/feature/'

if not os.path.exists(file_path):
    os.makedirs(file_path)


model_type = 'arousal'
# model_type = 'valence'

timestamp = datetime.datetime.now().strftime('%Y-%m-%d.%H.%M')  + '_' + model_type # 2018-04-21.15.59
log_path = os.path.join("runs", timestamp)



# n_input = 20*2048 # 20*2048
batch_size = 32*2
num_layers = 3
num_units = 256*2
dropout_rate = 0.5
num_epochs = 150

learning_rate_default = 0.0005
decay_steps = 10*(2440//batch_size)




def train(batch_size=batch_size, num_epochs=5, learning_rate_default=0.0001, Optimizer=tf.train.AdamOptimizer, reuse=False, trainable=True):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param inputs: input features
    :param labels: target labels
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :return: none
    """

    with tf.name_scope('Inputs'):
        input_var = tf.placeholder(tf.float32, [None, None, 2048], name='X_inputs')
        target_var = tf.placeholder(tf.float32, [None, 1], name='y_inputs')
        tf_is_training = tf.placeholder(tf.bool, None, name='is_training')
        tf_keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    X_train, y_train, X_val, y_val = load_feature_label(file_path)
    if model_type == 'arousal':
        y_train = y_train[:,0].reshape(-1,1)    # arousal
        y_val = y_val[:,0].reshape(-1,1)      # arousal
    elif model_type == 'valence':
        y_train = y_train[:,1].reshape(-1,1)    # valence
        y_val = y_val[:,1].reshape(-1,1)      # valence

    print('The shape of X_trian:\t', X_train.shape)
    print('The shape of X_val:\t', X_val.shape)
    print('The shape of y_trian:\t', y_train.shape)
    
    
    print("Building model and compiling functions...")
    prediction = build_lstm(X_input=input_var, batch_size=batch_size, num_layers=num_layers, num_units=num_units,
                        dropout_rate=tf_keep_prob, reuse=reuse, train=tf_is_training)

    Train_vars = tf.trainable_variables()


    with tf.name_scope('Loss'):
        _loss = tf.losses.mean_squared_error(labels=target_var, predictions=prediction)

    with tf.name_scope('Optimizer'):
        # learning_rate = learning_rate_default * Decay_rate^(global_steps/decay_steps)
        global_steps = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(
            learning_rate_default,  # Base learning rate.
            global_steps,
            decay_steps,
            0.95,  # Decay rate.
            staircase=True)
        optimizer = Optimizer(learning_rate)    # GradientDescentOptimizer  AdamOptimizer
        train_op = optimizer.minimize(_loss, global_step=global_steps, var_list=Train_vars)



    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, log_path))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss, mse and learning_rate
    loss_summary = tf.summary.scalar('loss', _loss)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, lr_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph()) # sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, tf.get_default_graph())


    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = checkpoint_dir + '/model'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    print("Starting training...")
    total_start_time = time.time()
    best_validation_ccc = 0

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        train_mse_epoch = [];   train_ccc_epoch = []
        val_ccc_epoch = [];     val_mse_epoch = []

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets = batch
                summary, _, pred, loss = sess.run([train_summary_op, train_op, prediction, _loss], 
                    {input_var: inputs, target_var: targets, tf_is_training: True, tf_keep_prob: dropout_rate})
                # print('\t\t\t', ccc(targets.reshape(-1), pred.reshape(-1)))
                
                train_err += loss
                train_batches += 1
                train_summary_writer.add_summary(summary, sess.run(global_steps))

            av_train_err = train_err / train_batches
            train_mse_epoch.append(av_train_err)


            dev_err  = dev_batches = 0
            val_pre = []; val_y = []
            for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                inputs, targets = batch
                summary, pred, loss = sess.run([dev_summary_op, prediction, _loss], 
                    {input_var: inputs, target_var: targets, tf_is_training: False, tf_keep_prob: 1.0})
                # print('\t\t\t', ccc(targets.reshape(-1), pred.reshape(-1)))
                dev_err += loss
                dev_batches += 1
                dev_summary_writer.add_summary(summary, sess.run(global_steps))
                val_pre.extend(pred.reshape(-1))
                val_y.extend(targets.reshape(-1))

            av_val_err = dev_err / dev_batches
            # ccc_val_valence = ccc(y_val[:len(val_pre)].reshape(-1), val_pre)
            ccc_val_valence = ccc(val_y, val_pre)
            val_mse_epoch.append(av_val_err)
            val_ccc_epoch.append(ccc_val_valence)


            if epoch % 10 == 0:
                train_pre = []; train_y = []
                for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                    inputs, targets = batch
                    pred = sess.run(prediction, {input_var: inputs, target_var: targets, tf_is_training: False, tf_keep_prob: 1.0})
                    train_pre.extend(pred.reshape(-1))
                    train_y.extend(targets.reshape(-1))
                # ccc_train_valence = ccc(y_train.reshape(-1), train_pre[:len(y_train)])
                ccc_train_valence = ccc(train_y, train_pre)
                train_ccc_epoch.append(ccc_train_valence)


            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(av_train_err))
            print("  validation loss:\t\t\t{:.6f}".format(av_val_err))
            print("  validation ccc_val:\t\t\t\t{:.6f}".format(ccc_val_valence))
            if epoch % 10 == 0: 
                print('-'*100)
                print("  train ccc_train:\t\t\t\t{:.6f}".format(ccc_train_valence))


            if ccc_val_valence > best_validation_ccc:
                best_validation_ccc = ccc_val_valence
                saver.save(sess, checkpoint_prefix)#, global_step=sess.run(global_steps))



        train_pre = []; train_y = []
        train_loss = []
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
            inputs, targets = batch
            loss, pred= sess.run([_loss, prediction], {input_var: inputs, target_var: targets, tf_is_training: False, tf_keep_prob: 1.0})
            train_loss.extend(loss.reshape(-1))
            train_pre.extend(pred.reshape(-1))
            train_y.extend(targets.reshape(-1))
        
        last_train_loss = np.mean(train_loss)
        ccc_train_valence = ccc(train_y, train_pre)
        # ccc_train_valence = ccc(y_train.reshape(-1), train_pre[:len(y_train)])


        last_val_loss = av_val_err
        print('-'*50)
        print('Time in total:', time.time()-total_start_time)
        print("Best validation ccc:\t\t\t{:.6f}".format(best_validation_ccc))
        
        print('-'*50)
        print("Last train mse:\t\t{:.6f}".format(last_train_loss))
        print("Last validation mse:\t\t{:.6f}".format(last_val_loss))
        print("Last train ccc_trian:\t\t\t{:.6f}".format(ccc_train_valence))
        

    train_summary_writer.close()
    dev_summary_writer.close()
    
    plt.subplot(2,2,1)
    plt.plot(train_mse_epoch, label='trian_mse')
    plt.plot(val_mse_epoch, label='val_mse')
    plt.plot(val_mse_epoch, '', label='val_mse')
    plt.legend()

    plt.subplot(2,2,2)
    x_indecs = [i for i in range(1,num_epochs+1,10)]
    plt.plot(x_indecs,train_ccc_epoch, label='train_ccc')
    plt.plot(val_ccc_epoch, label='val_ccc')
    plt.plot(val_ccc_epoch, label='val_ccc')
    plt.legend()

    plt.subplot(2,2,4)
    y_val = y_val.reshape(-1)[:len(val_pre)]
    val_pre = np.array(val_pre)
    idx_val = np.argsort(y_val)
    plt.plot(y_val[idx_val], label='y_val')
    plt.plot(np.array(val_pre)[idx_val], label='val_pre')
    plt.legend()

    # plt.savefig('./learning_cruve.png')
    plt.show()
    



def train_model(num_epochs=200):
    print('*'*200)
    print('\t\t Training the ' + 'cnn' + ' Model...')
    train(batch_size=batch_size, num_epochs=num_epochs,
                        learning_rate_default=0.0001, Optimizer=tf.train.AdamOptimizer)
    print('Done!')
    print('LSTM layers: ',num_layers, '\tunits: ', num_units, '\tdropout_rate: ', dropout_rate)

train_model(num_epochs=num_epochs)
# ['1dconv', 'maxpool', 'lstm', 'mix', 'cnn']
