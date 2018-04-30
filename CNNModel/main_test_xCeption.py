import os
import scipy.io as sp
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import tqdm
import numpy as np
from Models import FC

# parameters
FACE_INPUT_SHAPE = (80, 80, 3)
ORI_INPUT_SHAPE = (299, 299, 3)
INPUT_SHAPES = [FACE_INPUT_SHAPE, ORI_INPUT_SHAPE]
LABEL_NAMES = ['arousal', 'valence']
test_csv = 'results/omg_TestVideos_WithoutLabels.csv' # test csv
# loading csv
f = open(test_csv)
reader = csv.reader(f)
head = next(reader)
video_idx = head.index('video')
utt_idx = head.index('utterance')
print('INFO: load csv files...')
roots = [
    '/mnt/disk1/omg_competition/omg_Test_TMP/face_test',  # face video
         '/mnt/disk1/omg_competition/omg_Test_TMP/ori_test']  # ori video

multi_X = []

# for val
# loading data
print('INFO: Using xception extract features...')
paths = []
for row in tqdm.tqdm(reader):
    video = row[video_idx]
    utt = row[utt_idx]
    path = video + '-' + utt[:-4] + '.npz'
    paths.append(path)
print('INFO: Total video number:{}'.format(len(paths)))

for i in range(len(roots)):
    root = roots[i]
    INPUT_SHAPE = INPUT_SHAPES[i]
    test_X = []
    print(
        'INFO: Loading test data from {}, resize_shape:{}'.format(
            root,
            INPUT_SHAPE))
    # v2f = video2feature(INPUT_SHAPE)
    for p in tqdm.tqdm(paths):
        path = os.path.join(root, p)
        try:
            npf = np.load(path)
            features = npf['X']
        except BaseException:
            features = np.zeros(features.shape)
            print('ERROR INFO! {} doesn\'t exit'.format(path))
        test_X.append(features)
    test_X = np.array(test_X)
    print(test_X.shape)
    multi_X.append(test_X)


for i in range(len(LABEL_NAMES)):
    LABEL_NAME = LABEL_NAMES[i]
    results = []
    # ------------face video only------------------

    net = FC(multi_X[0].shape[1:], label_name=LABEL_NAME)
    net.load_weights('weights/ccc_best_face_1024_{}.h5'.format(LABEL_NAME))
    y_pred1 = net.predict(multi_X[0])
    results.append(y_pred1)
    # ------------original video only------------------
    print(multi_X[1].shape[1:])
    net = FC(multi_X[1].shape[1:], label_name=LABEL_NAME)
    X_test = np.array(multi_X[1])
    print(X_test.shape)
    net.load_weights('weights/ori_ccc_best_{}.h5'.format(LABEL_NAME))
    y_pred2 = net.predict(X_test)
    results.append(y_pred2)
    results = np.array(results)
    results = np.squeeze(results)
    if LABEL_NAME == 'arousal':
        arousal_res = results
    elif LABEL_NAME == 'valence':
        valence_res = results
    else:
        print("Error INFO! Error when saving files")
        exit(1)
    # break
sp.savemat('results/result_CNN_test_all.mat', {'arousal': arousal_res.T, 'valence': valence_res.T})
