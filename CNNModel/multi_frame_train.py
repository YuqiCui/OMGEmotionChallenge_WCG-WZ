# -*- coding: UTF-8 -*-
from Models import model_xception, FC, mul_FC
import keras.losses as loss
import keras.optimizers as optimizer
import calculateEvaluationCCC as CCC
import DataProcessor as DP
import keras.backend as K
from MyCallBack import mul_ccc_savebest as cb
from keras.callbacks import ReduceLROnPlateau
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LABEL_FLAG = 'valence'

def tensor_pearsonr(y_pred, y_true):
    mx = K.mean(y_pred)
    my = K.mean(y_true)
    xm = y_pred - mx
    ym = y_true - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = r_num / r_den
    return K.maximum(K.minimum(r, 1.0), -1.0)

def metric_CCC(y_true, y_pred):
    y_true = y_true
    y_pred=y_pred
    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)
    rho = tensor_pearsonr(y_pred, y_true)
    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)
    ccc = 2 * rho * std_gt * std_predictions / (
                K.square(std_predictions) + K.square(std_gt) + K.square(pred_mean - true_mean))
    return  ccc

FACE_SIZE = (80, 80, 3)
BATCH_SIZE = 64
EPOCHES = 5000

# load the data
data_root = '../each_frame_xception/{}'
ori_data_root = '../ori_each_frame_xception/{}'
train_X, train_arousal, train_valence = DP.load_data_multi([data_root.format('train'),ori_data_root.format('train')])
val_X, val_arousal, val_valence = DP.load_data_multi([data_root.format('val'),ori_data_root.format('val')])
print('success loading: train_X:{}, val_X:{}, arousal_train:{}, arousal_val:{}'.format(len(train_X),len(val_X),train_arousal.shape,val_arousal.shape))





sgd = optimizer.Adadelta()
# load the model
model = mul_FC(input_shape=[(2048,),(2048,)], label_name=LABEL_FLAG,num_frame=2)
# model.compile(loss=loss.mean_squared_error, optimizer=optimizer.Adadelta(), metrics=[metric_CCC])
model.compile(loss=loss.mean_squared_error, optimizer=optimizer.Adadelta(), metrics=[metric_CCC])
if LABEL_FLAG=='valence':
    train_y = train_valence
    val_y = val_valence
elif LABEL_FLAG=='arousal':
    train_y = train_arousal
    val_y = val_arousal
else:
    print('error: No available labels!')
    exit()

# Train--------------------------------------
# model.load_weights('weights/mul_ccc_best.h5')
# y_pre = model.predict(val_X, batch_size=1024)
# best_ccc = CCC.ccc(val_y,y_pre)
# ccc_callback=cb(label_flag=LABEL_FLAG,path='weights/mul_ccc_best.h5',best_score=best_ccc)
# model.fit(train_X,train_y,
#           batch_size=BATCH_SIZE,
#           validation_data=(val_X,val_y),
#           shuffle=True,
#           epochs=EPOCHES,
#           callbacks=[ccc_callback
#                      ])
# Predict--------------------------------------
print('val_CCC:')
model.load_weights('weights/mul_ccc_best.h5')
y_pre = model.predict(val_X, batch_size=1024)
score = CCC.ccc(val_y,y_pre)
print(score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
idx = numpy.argsort(val_y)
val = plt.plot(val_y[idx], 'r-')
plt.hold
true = plt.plot(y_pre[idx], 'g-')
plt.legend([val,true],['val','true'])
plt.savefig('pic_res/mul_{}_{}.jpg'.format(LABEL_FLAG,score))
