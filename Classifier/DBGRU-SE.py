import keras
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.model_selection import StratifiedKFold


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU, Bidirectional
from keras.models import Model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale

import utils.tools as utils



# from tensorflow.keras.layers import Input


def to_class(p):
    return np.argmax(p, axis=1)


def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


# Origanize data
def get_shuffle(dataset, label):
    # shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label


# Origanize data

data_ = pd.read_csv(r'SMOTEENN_GL_0.03.csv')
data = np.array(data_)
data = data[:,1:]
[m1, n1] = np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label = np.append(label1, label2)
X_ = scale(data)
y_ = label
X, y = get_shuffle(X_, y_)
sepscores = []
sepscores_ = []
ytest = np.ones((1,2))*0.5
yscore = np.ones((1,2))*0.5



def get_CNN_model(input_dim):
    gru = Sequential()
    gru.add(Bidirectional(GRU(int(input_dim/2), return_sequences=True),input_shape=(1,216)))
    gru.add(Dropout(0.5))
    gru.add(Bidirectional(GRU(int(input_dim/4), return_sequences=True)))
    gru.add(Dropout(0.5))
    return gru



def SE_block(input_dim):
    x = keras.layers.AvgPool1D(strides=1, padding="SAME", name="advage_pooling")(input_dim)
    temp_dim = x.shape[-1]
    x = Dense(units=temp_dim//4, use_bias=False, activation='relu')(x)
    x = Dense(units=temp_dim, use_bias=False, activation='hard_sigmoid')(x)
    output_se = keras.layers.Multiply()([input_dim,x])  
    return output_se




def model(input_dim, out_dim,sample_num):
    input_layer = keras.Input(shape=(None,input_dim))
    gru_out = get_CNN_model(input_dim)(input_layer)
    se_out = SE_block(gru_out)
    output1 = Dense(units = input_dim//4, activation='relu')(se_out)
    output2 = Dense(units = input_dim//8, activation='relu')(output1) 
    output = Dense(units=out_dim, activation='softmax', name="Dense_2")(output2)
    model = Model(input_layer,output)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


[sample_num, input_dim] = np.shape(X)
out_dim = 2
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
probas_rnn = []
tprs_rnn = []
sepscore_rnn = []
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(X, y):
    clf_rnn = model(input_dim, out_dim,sample_num)
    X_train_rnn = np.reshape(X[train], (-1, 1, input_dim))
    X_test_rnn = np.reshape(X[test], (-1, 1, input_dim))
    clf_list = clf_rnn.fit(X_train_rnn, to_categorical(y[train]).reshape((-1, 1, 2)), epochs=10)  # nb_epoch改为epochs
    y_rnn_probas = clf_rnn.predict(X_test_rnn)
    y_rnn_probas = y_rnn_probas.reshape((-1, 2))
    probas_rnn.append(y_rnn_probas)
    y_class = utils.categorical_probas_to_classes(y_rnn_probas)

    y_test = utils.to_categorical(y[test])  # generate the test
    ytest = np.vstack((ytest, y_test))
    y_test_tmp = y[test]
    yscore = np.vstack((yscore, y_rnn_probas))

    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class, y[test])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y[test], y_rnn_probas[:, 1])
    tprs_rnn.append(interp(mean_fpr, fpr, tpr))
    tprs_rnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_rnn.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])

row = ytest.shape[0]
ytest = ytest[np.array(range(1, row)), :]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_SE_GL_0.03.csv')

yscore_ = yscore[np.array(range(1, row)), :]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum_SE_GL_0.03.csv')

scores = np.array(sepscore_rnn)
result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscore_rnn.append(H1)
result = sepscore_rnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('SE__train_GL_0.03.csv')
