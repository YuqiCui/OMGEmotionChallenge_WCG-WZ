import numpy as np
import scipy.io as sp
import csv
from sklearn.metrics import mean_squared_error as mse

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
    'results/result_CNN.mat',
    'results/pred_valLabel.mat',
    'results/pre_wy.mat'
]
out_csv = 'results/omg_Validation_output.csv'
input_csv = 'results/omg_ValidationVideos.csv'


arousals = []
valences = []

for mat_file in mat_files:
    f = sp.loadmat(mat_file)
    if 'wy' in mat_file:
        arousals.append(f['arousal'].T)
        valences.append(f['valence'].T)
        # print(arousals[-1].shape)
        arousals[-1] = np.insert(arousals[-1],492,np.random.random([4,1]),axis=0)
        valences[-1] = np.insert(valences[-1],492, np.random.random([4, 1]), axis=0)
    else:
        arousals.append(f['arousal'])
        valences.append(f['valence'])
    # print(f['arousal'].shape)

arousal = np.concatenate((arousals), axis=1).T
valence = np.concatenate((valences), axis=1)
f = sp.loadmat('results/omg_pred_valence.mat')
tmp = f['valence'].T
tmp = np.insert(tmp,492,np.random.random([4,1]),axis=0)
valence = np.concatenate((valence,tmp),axis=1).T
# mean
# arousal_SMLR = np.mean(arousal,axis=0)
# valence_SMLR = np.mean(valence,axis=0)
# SMLR
# arousal = np.delete(arousal,1,axis=0) # remove CNN-video model oupputs
valence = np.delete(valence,3,axis=0)
arousal_SMLR = SMLR(arousal)
valence_SMLR = SMLR(valence)
from scipy.stats import pearsonr
def ccc(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = np.std(y_pred)

    std_gt = np.std(y_true)


    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc

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


# load label for val
f = sp.loadmat('label.mat')
arousal_label = f['arousal']
valence_label = f['valence']
from calculateEvaluationCCC import ccc
# compute ccc for each view
for i in range(arousal.shape[0]):
    print('view {}: Arousal score:{:.4f}'.format(
        i, ccc(arousal_label.ravel(), arousal[i, :].ravel())[0]))
for i in range(valence.shape[0]):
    print('view {}: Valence score:{:.4f}'.format(
        i, ccc(valence_label.ravel(), valence[i, :].ravel())[0]))
print('------------------------------------------')
for i in range(arousal.shape[0]):
    print('view {}: Arousal mse:{:.4f}'.format(
        i, mse(arousal_label.ravel(), arousal[i, :].ravel())))
for i in range(valence.shape[0]):
    print('view {}: Valence mse:{:.4f}'.format(
        i, mse(valence_label.ravel(), valence[i, :].ravel())))


print('Arousal score:{:.4f}'.format(ccc(arousal_label.ravel(), arousal_SMLR.ravel())[0]))
print('Valence score:{:.4f}'.format(ccc(valence_label.ravel(), valence_SMLR.ravel())[0]))
print('------------------------------------------')
print('Arousal mse:{:.4f}'.format(mse(arousal_label.ravel(), arousal_SMLR.ravel())))
print('Valence mse:{:.4f}'.format(mse(valence_label.ravel(), valence_SMLR.ravel())))