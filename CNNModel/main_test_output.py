import numpy as np
import scipy.io as sp
import csv


def SMLR(X):
    cov = np.cov(X)
    value, vector = np.linalg.eig(cov)
    max_id = np.argmax(value)
    weight = vector[:, max_id]
    weight = weight / sum(weight)
    X = np.dot(X.T, weight.reshape(-1, 1))
    return X


# result files
mat_files = [
    'results/result_CNN_test_all.mat',
    'results/pred_TestLabel.mat',
    'results/test_pre_wy.mat'
]
out_csv = 'results/omg_TestVideo_Output.csv'
input_csv = 'results/omg_TestVideos_WithoutLabels.csv'


arousals = []
valences = []

for mat_file in mat_files:
    f = sp.loadmat(mat_file)
    if 'wy' in mat_file:
        arousals.append(f['arousal'].T)
        valences.append(f['valence'].T)
    else:
        arousals.append(f['arousal'])
        valences.append(f['valence'])
    print(f['arousal'].shape)

arousal = np.concatenate((arousals), axis=1).T
valence = np.concatenate((valences), axis=1).T

# mean
# arousal_SMLR = np.mean(arousal,axis=0)
# valence_SMLR = np.mean(valence,axis=0)
# SMLR
arousal = np.delete(arousal, 1, axis=0)  # remove CNN-video model oupputs
arousal_SMLR = SMLR(arousal)
valence_SMLR = SMLR(valence)


# output to csv

f_read = open(input_csv)
reader = csv.reader(f_read)
header = next(reader)
header.append('arousal')
header.append('valence')
print(header)
read_rows = []
count = 0
for row in reader:
    row.extend(arousal_SMLR[count])
    row.extend(valence_SMLR[count])
    read_rows.append(row)
    count += 1
f_write = open(out_csv, 'w+', newline='')
writer = csv.writer(f_write)
writer.writerow(header)
for row in read_rows:
    writer.writerow(row)
f_read.close()
f_write.close()


