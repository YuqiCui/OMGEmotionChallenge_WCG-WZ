
import cv2
import csv
import os
import numpy as np
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.applications.xception import Xception, preprocess_input

# define global variable
csv_file = ['/mnt/disk1/omg_competition/submmit_version/results/omg_TestVideos_WithoutLabels.csv']
face_video_root = [
    '/mnt/disk1/omg_competition/omg_Test_faces']
net = Xception(include_top=False, input_shape=(80, 80, 3), pooling='max')
verbose = 0
mode = ['test']
savepath = '/mnt/disk1/omg_competition/omg_Test_TMP/face_{}/{}.npz'


def feature_extraction(cap):
    frames = []
    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(
            frame, (80, 80), interpolation=cv2.INTER_NEAREST).astype(float)
        frames.append(frame)
    frame_num = len(frames)
    if len(frames) == 0:
        return None, frame_num
    input = np.array(frames).astype(float)
    input = preprocess_input(input)
    output = net.predict(input)
    output = np.mean(output,axis=0)
    # print(output.shape)
    # import sys
    # sys.exit(0)
    return output, frame_num


if __name__ == "__main__":
    idx = 0
    for file in csv_file:
        f = open(file)
        reader = csv.reader(f)
        next(f)
        csv_count = 0
        frame_nums = []
        for row in tqdm.tqdm(reader):
            p = row[3] + '/' + row[4][:-3] + 'avi'
            if mode[idx] != 'test':
                arousal = row[5]
                valence = row[6]
                label = np.array([arousal, valence]).astype(float)
            # load AVI file
            video = os.path.join(face_video_root[idx], p)
            if not os.path.exists(video):
                print('Error! Video {} doesn\'t exist'.format(video))
                continue
            cap = cv2.VideoCapture(video)
            features, num = feature_extraction(cap)
            if verbose == 1:
                print(
                    'Num:{}\tLoading video from {}, frame num:{}'.format(
                        csv_count, video, num))
            else:
                pass
            if features is None:
                print('Error! Video {} return None!'.format(video))
                continue
            if mode[idx] != 'test':
                np.savez(savepath.format(mode[idx], row[3] + '-' + row[4][:-4]),
                         X=features, y=label, video=row[3], utterance=row[4])
            else:
                np.savez(savepath.format(mode[idx], row[3] + '-' + row[4][:-4]),
                         X=features, video=row[3], utterance=row[4])
            csv_count += 1
            # break
        # np.savez('data/each_frame_num_{}.npz'.format(mode[idx]), data=np.array(frame_nums))
        idx += 1
