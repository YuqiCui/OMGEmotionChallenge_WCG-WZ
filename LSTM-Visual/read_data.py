#coding:utf-8
import numpy as np
import scipy.io
import os
import csv
import imageio
from PIL import Image

train_file = './set_csv/wy_omg_TrainVideos.csv'
train_root = '/mnt/disk1/omg_competition/omg_Train/'

val_file = './set_csv/wy_omg_ValidationVideos.csv'
val_root = '/mnt/disk1/omg_competition/omg_Val/'

test_file = './set_csv/omg_TestVideos_WithoutLabels.csv'
test_root = '/mnt/disk1/omg_competition/omg_Test_TMP/omg_Test'

feature_path = '/mnt/disk2/ywang/omg_competition_images/feature/'

filename = '/mnt/disk1/omg_competition/omg_Train/04899849f_1/utterance_1.mp4'


class get_file():
    def __init__(self,train_file=train_file, train_root=train_root, val_file=val_file, val_root=val_root, test_file=test_file, test_root=test_root):
        self.train_root = train_root
        self.val_root = val_root
        self.test_root = test_root

        self.train_csv = csv.reader(open(train_file))
        self.val_csv  = csv.reader(open(val_file))
        self.test_csv  = csv.reader(open(test_file))
        
        self.set_list()
       
    def set_list(self):
        train_list = []
        train_path = []
        train_lables = []
        j = 0
        for row in self.train_csv:
            if j==0:
                j = j + 1
                continue
            name, utterance, arousal, valence, EmotionMaxVote=row[3:]
            file_path = os.path.join(train_root, name, utterance)
            arousal, valence, EmotionMaxVote = map(float, [arousal, valence, EmotionMaxVote])
            
            train_path.append(file_path)
            train_lables.append([arousal, valence, EmotionMaxVote])
            train_list.append([file_path, arousal, valence, EmotionMaxVote])        # train_list = np.array(train_list)     # 转成array后，list内的float类型也转换成了str
            # [['/mnt/disk1/omg_competition/omg_Train/5b44393ed/utterance_4.mp4', 0.17009126740299998, -0.0236966881471, 4.0]]
        
        self.train_list = train_list
        self.train_path = train_path
        self.train_lables = np.array(train_lables)
        print("trian_lise's shape:\t", np.array(train_list).shape)     # (2440, 4)

        val_list = []
        val_path = []
        val_lables = []
        j = 0
        for row in self.val_csv:
            if j==0:
                j = j + 1
                continue
            name, utterance, arousal, valence, EmotionMaxVote=row[3:]
            file_path = os.path.join(val_root, name, utterance)
            arousal, valence, EmotionMaxVote = map(float, [arousal, valence, EmotionMaxVote])
            val_path.append(file_path)
            val_lables.append([arousal, valence, EmotionMaxVote])
            val_list.append([file_path, arousal, valence, EmotionMaxVote])
        self.val_list = val_list
        self.val_path = val_path
        self.val_lables = np.array(val_lables)
        print("val_lise's shape:\t", np.array(val_list).shape)         # (617, 4)


        test_list = []
        test_path = []
        j = 0
        for row in self.test_csv:
            if j==0:
                j = j + 1
                continue
            name, utterance = row[3:]
            file_path = os.path.join(test_root, name, utterance)
            test_path.append(file_path)
            test_list.append([file_path])
        self.test_list = test_list
        self.test_path = test_path
        print("test_lise's shape:\t", np.array(test_list).shape)


def extract_images(filename=filename):
    # extract frames from videos
    save_dir = '/mnt/disk2/ywang/omg_competition_images/'
    if not os.path.exists(filename):
        print('No such file:\t', filename)
        return 200
    else:
        if not os.path.exists(save_dir + filename[27:-4]):
            os.makedirs(save_dir + filename[27:-4])

    vid = imageio.get_reader(filename, 'ffmpeg')# , size='229x229')
    N = vid.get_length() - 1   # get the numbers of image in the file
    print('****************************',N)
    if N < 20:
        for i in range(21-N):
            print('\t',i)
            temp = vid.get_data(0)
            image = Image.fromarray(temp)
            image.save(save_dir + filename[27:-4] + '/' + str(i) +'.png')
        print('----------------------------------')
        for i in range(21-N, 20):
            print('\t',i)
            print('\t\t', i-20+N)
            temp = vid.get_data(i-20+N)
            image = Image.fromarray(temp)
            image.save(save_dir + filename[27:-4] + '/' + str(i) +'.png')

    else:
        stride = N//20
        print('\t\t\t\t',stride)
        for i in range(20):
            print('\t\t\t\t\t', i*stride)
            temp = vid.get_data(i*stride)
            image = Image.fromarray(temp)
            image.save(save_dir + filename[27:-4] + '/' + str(i) +'.png')


def get_frames(model='Train'):
    file = get_file(train_file, train_root, val_file, val_root, test_file, test_root)
    if model is 'Train':
        path = file.train_list
    elif model is 'Val':
        path = file.val_list
    elif model is 'Test':
        path = file.test_list
    
    for i in range(len(np.array(path))):
        print(i)
        extract_images(path[i][0])


def get_images_path(model='Train'):
    save_dir = '/mnt/disk2/ywang/omg_competition_images/'
    file = get_file(train_file, train_root, val_file, val_root, test_file, test_root)
    if model is 'Train':
        path = file.train_path
    elif model is 'Val':
        path = file.val_path
    elif model is 'Test':
        path = file.test_path
    # '/mnt/disk1/omg_competition/omg_Train/5b44393ed/utterance_4.mp4'
    # '/mnt/disk1/omg_competition/omg_Val/c0f84b343_2/utterance_13.mp4'
    # filename[27:-4]   即 omg_Train/5b44393ed/utterance_4
    #                   或 omg_Val/c0f84b343_2/utterance_13
    
    N = len(path)   # numbers of samples
    images_path = []
    for i in range(N):
        for j in range(20):
            temp = save_dir + path[i][27:-4] + '/' + str(j) + '.png'
            images_path.append(temp)
    images_path = np.array(images_path).reshape(N, 20)
    print('The shape of images_path:\t', images_path.shape)
    return images_path
    


def load_feature_label(feature_path=feature_path):
    print('-'*100)
    print("Loading feature and label from %s" % (feature_path))
    
    Train_feature =  scipy.io.loadmat(feature_path + 'Train_images_features.mat')['features']
    Train_label =  scipy.io.loadmat(feature_path + 'Train_labels.mat')['labels'][:,:2]
    Val_feature =  scipy.io.loadmat(feature_path + 'Val_images_features.mat')['features']
    Val_label =  scipy.io.loadmat(feature_path + 'Val_labels.mat')['labels'][:,:2]
    print("Train feature's shape:\t", Train_feature.shape)  # (2440, 20, 2048)
    print("Train label's shape:\t", Train_label.shape)      # (2440, 3)
    print("Val feature's shape:\t", Val_feature.shape)      # 617, 20, 2048)
    print("Val label's shape:\t", Val_label.shape)          # (617, 3)

    return Train_feature, Train_label, Val_feature, Val_label

def load_feature_label_test(feature_path=feature_path):
    print('-'*100)
    print("Loading test feature from %s" % (feature_path))
    
    Test_feature =  scipy.io.loadmat(feature_path + 'Test_images_features.mat')['features']
    print("Test feature's shape:\t", Test_feature.shape)  # (2440, 20, 2048)

    return Test_feature


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 3D numpy array for images [n_samples, n_timewindows, feature].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    
    input_len = inputs.shape[0]
    assert input_len == len(targets)

    if shuffle:
        indices = np.arange(input_len)  
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if start_idx + batchsize >= input_len:
            start_idx = input_len - batchsize
        
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



if __name__=="__main__":
    file = get_file(train_file, train_root, val_file, val_root, test_file, test_root)
    get_frames(model='Test')


    # scipy.io.savemat(save_path +'Train_labels.mat', {'labels': file.train_lables})
    # scipy.io.savemat(save_path +'Val_labels.mat', {'labels': file.val_lables})

