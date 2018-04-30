# -*- coding: UTF-8 -*-
"""
This file is to detect the face in the video and resizes it to (50,50). Then these faces will be saved as a .avi file.
"""
import cv2
import os
import face_recognition
from tqdm import tqdm

"""
the structure of files:
    data/omg_Val    (input_root_dir)
        |____adfaefadfefad  (input_path)
                |____1.mp4  (input_file)
"""


def video2facesAVI(input_root_dir, output_root_dir, input_list, face_size):
    count = 0
    for dir_name in input_list:
        count += 1
        print('Processing: {}/{}...'.format(count, len(input_list)))
        input_path = os.path.join(input_root_dir, dir_name)

        # make output dirs
        output_path = os.path.join(output_root_dir, dir_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Traversal folder
        if os.path.isdir(input_path):
            input_file_list = os.listdir(input_path)

            # Traversal files
            for file_name in tqdm(input_file_list):
                input_file = os.path.join(input_path, file_name)
                output_file = os.path.join(output_path, file_name)
                if os.path.isfile(input_file):

                    # Get the input movie and build an output movie.
                    input_movie = cv2.VideoCapture(input_file)
                    FPS = input_movie.get(cv2.CAP_PROP_FPS)

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_movie = cv2.VideoWriter(output_file[:-3] + 'avi', fourcc, FPS, face_size)

                    # Go through the Frames
                    while True:
                        ret, frame = input_movie.read()
                        if not ret:
                            break
                        # Detect the faces
                        temp = frame[:, :, ::-1]
                        face_locations = face_recognition.face_locations(temp, model='cnn')

                        if len(face_locations) == 0:
                            continue

                        (top, right, bottom, left) = face_locations[0]
                        face_frame = frame[top:bottom, left:right, :]
                        face_frame = cv2.resize(face_frame, face_size)
                        output_movie.write(face_frame)
                    input_movie.release()
                    output_movie.release()




input_root_dir = 'data/omg_Val'
output_root_dir = 'data/omg_Val_faces2'
face_size = (50, 50)

if not os.path.exists(input_root_dir):
    print('error: no such input_path!')
    exit(0)

input_list = os.listdir(input_root_dir)
video2facesAVI(input_root_dir, output_root_dir, input_list, face_size)
