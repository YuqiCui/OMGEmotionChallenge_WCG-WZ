# -*- coding: UTF-8 -*-
from Models import model_xception, FC
import keras.losses as loss
import keras.optimizers as optimizer
import calculateEvaluationCCC as CCC
import DataProcessor as DP
import keras.backend as K
from MyCallBack import ccc_savebest as cb
from MyCallBack import loss_history as lh
from algorithm.preprocess import preprocess
from keras.callbacks import EarlyStopping
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LABEL_FLAG = 'valence'
# savepath = 'weights/ccc_best_512_clip_pca_valOnVal_{}.h5'
savepath = 'weights/ccc_best_512_clip_pca_{}.h5'
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
EPOCHES = 3000

# load the data
data_root = '../each_frame_xception'
train_data_dir = os.path.join(data_root,'train')
val_data_dir = os.path.join(data_root,'val')
train_X, train_arousal, train_valence = DP.load_data(train_data_dir)
val_X, val_arousal, val_valence = DP.load_data(val_data_dir)
print('success loading: train_X:{}, val_X:{}'.format(train_X.shape,val_X.shape))
sgd = optimizer.Adadelta()
if LABEL_FLAG=='valence':
    train_y = train_valence
    val_y = val_valence
elif LABEL_FLAG=='arousal':
    train_y = train_arousal
    val_y = val_arousal
else:
    print('error: No available labels!')
    exit()
# preprocessing
X = preprocess(np.concatenate([train_X,val_X],axis=0),use_pca=True)
train_X = X[:train_X.shape[0]]
val_X = X[train_X.shape[0]:]
print(train_X.shape)
exit()

# model
# load the model
model = FC(input_shape=train_X.shape[1:], label_name=LABEL_FLAG)
model.compile(loss=loss.mean_squared_error, optimizer=sgd, metrics=[metric_CCC])
# callbacks
ccc_callback=cb(label_flag=LABEL_FLAG,path=savepath.format(LABEL_FLAG))
loss_history = lh(test_data=[val_X, val_y], label_flag=LABEL_FLAG)
earlyStop = EarlyStopping(monitor='val_metric_CCC',patience=200,verbose=1,mode='max')

# model.load_weights(savepath.format(LABEL_FLAG))
# y_pre = model.predict(val_X, batch_size=1024)
# best_ccc = CCC.ccc(val_y,y_pre)
model.fit(train_X,train_y,
          batch_size=BATCH_SIZE,
          # validation_data=(val_X,val_y),
          validation_split=0.2,
          shuffle=True,
          epochs=EPOCHES,
          callbacks=[ccc_callback,
                     loss_history,
                        earlyStop
                     ])
print('val_CCC:')
model.load_weights(savepath.format(LABEL_FLAG))
y_pre = model.predict(val_X, batch_size=1024)
score = CCC.ccc(val_y,y_pre)
print(score)

