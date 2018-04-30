# OMGEmotionChallenge_WCG-WZ
This is the code of WCG-WZ team for OMG Emotion Challenge.

## Method description
This method contains four parts:
  + CNN Face Model
  + CNN Visual Model
  + LSTM Visual Model
  + SVM Audio Model

## Face Extraction

We used the face *[recognition package](https://github.com/ageitgey/face_recognition)* to crop out the face of the actor in each frame of an utterance. We also used *[opencv library](https://opencv.org)* to do some simple image processing, e. g. rescaling images. Each face image was rescaled to 80 × 80 × 3 (height × width × channel). 

### Requirements

- **face recognition**: [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- **opencv**:[https://opencv.org/](https://opencv.org/ "https://opencv.org/")


## CNN Face Model
We extracted face features by Xception with weights pre-trained on ImageNet. Each utterance gave n 2048-d feature vectors, where n is the number of frames. We then took the **average** of these n 2048-d feature vectors to obtain a single 2048-d feature vector for each utterance. 
These features were next passed through a three-layer multi-layer perception (MLP) for regression. The hidden layer had 1024 nodes with ReLU activation, and the output layer had only one node with sigmoid activation for arousal, and linear activation for valence. Optimization of the MLP was done using Adamdelta, with dropout rate 0.25. The validation ccc was used to determine when the training

## CNN Visual Model
This model was almost identical to CNN-Face, except that the entire frame instead of only the face was used to extract the features.

## LSTM Visual Model
For each utterance, we down-sampled 20 frames uniformly in time (if an utterance had less than 20 frames, then the ﬁrst frame was repeated to make up 20 frames), and then used InceptionV3 pre-trained on ImageNet, to obtain a 20 × 2048 feature matrix. Next we applied multi-layer long short-term memory (LSTM) to extract the time domain information, and an MLP with 512 hidden nodes and one output node for regression. Dropout and ReLU activation were used in both LSTM and MLP.

## SVM Audio Model
The feature we extracted in audio is shown in the following table.

|  Feature category | Number  | Value  |
| ------------ | ------------ | ------------ |
| Spectral centroid  |  2 | Mean, variance  |
| Band energy radio  |  2 | Mean, variance  |
| Delta spectrum magnitude  |  2 |  Mean, variance |
| Zero crossing rate  | 2  |  Mean, variance |
| Pitch  | 2 | Mean, variance  |
| Short-time average energy  |  2 | Mean, variance  |
| Silence ratio  |  2 |  Mean |
| MFCC coefficients  |  24 | Mean, variance  |
| Delta MFCC  | 12  |   Mean|
| LPCC  |  22 |Mean, variance   |
| Formant  | 5  |  Mean |

Then we use relieff to select features and feed to SVR.

## Model fusion by SMLR
We use SMLR approach to combine all final output.
>D. Wu, V. J. Lawhern, S. Gordon, B. J. Lance, and C.-T. Lin,
“Spectral meta-learner for regression (SMLR) model aggrega-
tion: Towards calibrationless brain-computer interface (BCI),”
in Proc. IEEE Int’l Conf. on Systems, Man and Cybernetics,
Budapest, Hungary, October 2016, pp. 743–749.
