import keras
import calculateEvaluationCCC as CCC
class loss_history(keras.callbacks.Callback):
    def __init__(self, test_data, label_flag):
        self.label_flag = label_flag
        self.x = test_data[0]
        self.y = test_data[1]
    def on_train_begin(self, logs=None):
        self.train_loss = []
        self.val_loss = []
        self.val_ccc = []
        self.test_ccc = []
    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x)
        score = CCC.ccc(self.y, pred)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_ccc.append(logs.get('val_metric_CCC'))
        self.test_ccc.append(score)
    def loss_plot(self):
        iters = range(len(self.train_loss))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(iters, self.train_loss, 'r', label='train loss')
        ax1.plot(iters, self.val_loss, 'g', label='val loss')
        ax1.set_ylabel('mse loss')
        ax1.set_xlabel('epoch')
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(iters, self.val_ccc, 'b', label='val ccc' )
        ax2.plot(iters, self.test_ccc, 'k', label='test ccc' )
        ax2.set_ylabel('ccc')
        ax2.legend(loc="upper right")
        plt.title(self.label_flag)
        plt.show()

class ccc_savebest(keras.callbacks.Callback):
    def __init__(
            self,
            path='weights/ccc_best.h5',
            label_flag=None,
            verbose=1):
        self.label_flag = label_flag
        self.best_ccc = -1
        self.path = path
        self.verbose = verbose
        self.count = 0


    def on_epoch_end(self, epoch, logs=None):
        x = self.validation_data[0]
        y = self.validation_data[1]
        # print(y)
        pred = self.model.predict(x)
        score = CCC.ccc(y, pred)
        if score >= self.best_ccc:
            self.model.save(self.path, overwrite=True)
            self.best_ccc = score
            print(' - Epoch %05d: ccc improved from %.5f to %.5f,'
                  ' saving model to %s'
                  % (epoch, self.best_ccc, score, self.path))
            if self.verbose == 1:
                print(
                    ' - Epoch %05d: saving model to %s, ccc:%.5f' %
                    (epoch, self.path, score))
        else:
            print(' - CC not improved,{:.5}(best:{:.5})'.format(score,self.best_ccc))

class mul_ccc_savebest(keras.callbacks.Callback):
    def __init__(
            self,
            path='weights/ccc_best.h5',
            label_flag=None,
            verbose=1,
            length=2,
            best_score=-1):
        self.label_flag = label_flag
        self.best_ccc = best_score=-1
        self.path = path
        self.verbose = verbose
        self.count = 0
        self.length=length


    def on_epoch_end(self, epoch, logs=None):
        x = self.validation_data[:self.length]
        y = self.validation_data[self.length]
        # print(y)
        pred = self.model.predict(x)
        score = CCC.ccc(y, pred)
        if score >= self.best_ccc:
            self.model.save(self.path, overwrite=True)
            self.best_ccc = score
            print(' - Epoch %05d: ccc improved from %.5f to %.5f,'
                  ' saving model to %s'
                  % (epoch, self.best_ccc, score, self.path))
            if self.verbose == 1:
                print(
                    ' - Epoch %05d: saving model to %s, ccc:%.5f' %
                    (epoch, self.path, score))
        else:
            print(' - CC not improved,{:.5}(best:{:.5})'.format(score,self.best_ccc))

