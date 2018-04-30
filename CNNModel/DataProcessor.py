# -*- coding: UTF-8
import csv
import numpy as np
import os
import cv2
FACE_SIZE = (80, 80)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def load_data(root_dir):
    file_list = os.listdir(root_dir)
    X = []
    arousals = []
    valences = []
    for file in file_list:
        path = os.path.join(root_dir,file)
        if os.path.isfile(path):
            temp = np.load(path)
            temp_X = temp['X']
            temp_label = temp['y']
            arousals.append(float(temp_label[0]))
            valences.append(float(temp_label[1]))
            temp_X = np.mean(temp_X,axis=0,keepdims=False)
            X.append(temp_X)
    return [np.array(X), np.array(arousals), np.array(valences)]

def load_data_multi(root_dirs):
    """
    :param root_dirs: make sure face root is at the index 0, cause ori can't be missed, but face can
    :return:
    """
    padding_shape=(2048,)
    num_dim=len(root_dirs)
    file_list=[]
    data={}
    dim = 0
    for root in root_dirs:
        files = os.listdir(root)
        for file in files:
            path = os.path.join(root,file)
            temp = np.load(path)
            temp_X = temp['X']
            temp_label = temp['y']
            temp_X = np.mean(temp_X, axis=0, keepdims=False)
            data[file] = ([np.array(temp_X)],np.array(temp_label[0]).astype(float),np.array(temp_label[1]).astype(float))
            if dim != 0 and file not in data_multi:
                padding_x = []
                for i in range(dim):
                    padding_x.append(np.zeros(padding_shape))
                padding_x.append(temp_X)
                data_multi[file]=(padding_x,np.array(temp_label[0]).astype(float),np.array(temp_label[1]).astype(float))
            elif dim!=0 and file in data_multi:
                a,b,c = data_multi[file]
                a.append(temp_X)
                data_multi[file]=(a,b,c)

        if dim == 0:
            data_multi = data
        dim += 1

    x = [[] for i in range(num_dim)]
    arousals = []
    valences = []
    for k,v in data_multi.items():
        a,b,c=v
        assert(len(a)==num_dim),'Error on {}!'.format(k)
        for i in range(num_dim):
            x[i].append(a[i])
        arousals.append(b)
        valences.append(c)
    for i in range(num_dim):
        x[i]=np.array(x[i])
    return x, np.array(arousals), np.array(valences)

def save_label_dict(reading_file):
    dict_arousals = {}
    dict_valences = {}
    with open(reading_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rowNumber = 0
        for row in spamreader:
            if rowNumber > 0:
                video = str(row[3])
                utterance = str(row[4])
                arousal = float(row[5])
                valence = float(row[6])
                name = video+'/'+utterance
                dict_arousals[name] = arousal
                dict_valences[name] = valence
            rowNumber += 1
    return (dict_arousals, dict_valences)



