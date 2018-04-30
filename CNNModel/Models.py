# -*- coding: UTF-8 -*-
from keras.applications.xception import Xception
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input,Dropout,Merge,Average



def FC(input_shape, label_name):
    inputs = Input(shape=input_shape)
    x = Dropout(0.2)(inputs)
    x = Dense(1024, activation='relu')(x)
    if label_name == 'arousal':
        output = Dense(units=1, activation='sigmoid')(x)
    elif label_name == 'valence':
        output = Dense(units=1, activation='linear')(x)

    # build the model.
    model = Model(inputs=inputs, outputs=output)
    return model
def mul_FC(input_shape,label_name,num_frame):
    input = []
    for i in range(num_frame):
        model = Sequential()
        model.add(Dropout(0.25,input_shape=input_shape[i]))
        model.add(Dense(512,activation='relu'))
        input.append(model)
    avg_model = Merge(input,mode='concat')
    model_all = Sequential()
    model_all.add(avg_model)
    # model_all.add(Dropout(0.2))
    if label_name == 'arousal':
        model_all.add(Dense(1,activation='sigmoid'))
    elif label_name == 'valence':
        model_all.add(Dense(1, activation='linear'))

    return model_all

