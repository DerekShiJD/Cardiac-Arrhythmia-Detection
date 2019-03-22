import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt

from ecg import testload
from ecg import result_analysis as ana

Type = ['Sinus', 'Paced', 'Rbbb']

# -------------------------------------------------------------
# load data

Input_Shape = (200, 1)

'''
X_test, Y_test, classes = testload.load_dataset(Input_Shape)
np.save('data/load_testdata/X_test.npy', X_test)
np.save('data/load_testdata/Y_test.npy', Y_test)
np.save('data/load_testdata/classes.npy', classes)
'''

X_test = np.load('data/load_testdata/X_test.npy')
Y_test = np.load('data/load_testdata/Y_test.npy')
classes = np.load('data/load_testdata/classes.npy')

print("number of test examples = " + str(X_test.shape[0]))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# -------------------------------------------------------------

model = load_model('models/SimpleModel_v7.h5')

preds = model.evaluate(x=X_test, y=Y_test)

print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

# -------------------------------------------------------------
# ROC AUC plot
Y_score = model.predict(X_test)

for i in range(Y_score.shape[1]):
    ana.plot_roc(Y_test, Y_score, i, Type)
    ana.plot_auc2(Y_test, Y_score, i, Type)

# Confusion matrix
ana.plot_confusion_matrix(Y_test, Y_score, Type, 'Confusion Matrix')

# -------------------------------------------------------------
# visualize the prediction process

Type = ['Sinus', 'Paced', 'Rbbb']


# wrong predictions
'''
length = Y_test.shape[0]

for i in range(length):
    predlist = [0,0,0]
    answer = [0,0,0]
    pred = model.predict(X_test[i].reshape(1, 200, 1))
    result = max(pred)
    for j in range(3):
        maxnumber = max(pred[0])
        if maxnumber == pred[0][j]:
            predlist[j] = 1
            break
    for j in range(3):
        if Y_test[i][j] == 1:
            answer[j] = 1
            break
    if predlist != answer:
        print('wrong predict: ' + str(predlist[0]) + str(predlist[1]) + str(predlist[2]))
        print('right answer:' + str(answer[0]) + str(answer[1]) + str(answer[2]))
'''

# -------------------------------------------------------------
# display outcomes
'''
for i in range(20):
    x_pred = np.array(X_test[i])
    x_pred = x_pred.reshape(1, 200, 1)
    x_print = X_test[i]
    y_pred = Y_test[i]

    yr = model.predict(x_pred)

    max = yr[0][0]
    id = 0

    for j in range(2):
        if yr[0][j+1] > max:
            max = yr[0][j+1]
            id = j + 1

    for k in range(3):
        if y_pred[k] == 1:
            real = Type[k]
            break

    result = Type[id]

    #print('Beat type: ' + real + ' Prediction:  ' + result)
    plt.plot(x_print)
    plt.title('CNN-based Cardiac Arrhythmia Detection\nBeat type: ' + real + '\nPrediction:  ' + result)
    plt.show()
'''



