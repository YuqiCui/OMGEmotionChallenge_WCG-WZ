#coding:utf-8

import os
import scipy.io
import numpy as np
import tensorflow as tf
from read_data import get_images_path



os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# the filepath for InceptionV3
MODEL_DIR = './inception-2015-12-05'
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

# the model name of the pretrained InceptuonV3
MODEL_FILe = 'classify_image_graph_def.pb'

BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'



def inception_feature(model='Train'):   # model is 'Train' or 'Val'
    images_path = get_images_path(model)
    save_path = '/mnt/disk2/ywang/omg_competition_images/feature/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    inception_graph_def_file = os.path.join(MODEL_DIR, MODEL_FILe)
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor_name, jepg_data_tensor_name = tf.import_graph_def(graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    
    with tf.Session() as sess:
        n_samples, n_frames = images_path.shape
        features = np.zeros((n_samples, n_frames, BOTTLENECK_TENSOR_SIZE))
        
        for i in range(n_samples):
            for j in range(n_frames):
                print(i,'-', j, '\t\t', images_path[i][j])
                image_data = tf.gfile.FastGFile(images_path[i][j], 'rb').read()
        
                bottleneck_tensor_value = np.squeeze(
                        sess.run(bottleneck_tensor_name, {jepg_data_tensor_name: image_data}))
                features[i][j] = bottleneck_tensor_value
                print('\t', bottleneck_tensor_value)

        
        scipy.io.savemat( save_path + model +'_images_features.mat', {'features': features})
        print('Computes inception features done!')
        print('Saving path:\t', save_path + model +'_images_features.mat')



if __name__=="__main__":
    # inception_feature('Train')
    # inception_feature('Val')
    inception_feature('Test')