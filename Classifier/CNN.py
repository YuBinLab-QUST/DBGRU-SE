import numpy as np
import pandas as pd
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import scale
import utils
import matplotlib.pyplot as plt


# Origanize data

data_=pd.read_csv(r'SMOTEENN_GL_0.03.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5


def get_CNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    model.add(Conv1D(filters = 32, kernel_size =  3, padding = 'same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    #model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))  
    model.add(Flatten())
    model.add(Dense(int(input_dim/4), activation = 'relu'))
    model.add(Dense(int(input_dim/8), activation = 'relu'))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    return model

[sample_num,input_dim]=np.shape(X)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y):
    clf_cnn = get_CNN_model(input_dim,out_dim)
    X_train_cnn=np.reshape(X[train],(-1,1,input_dim))
    X_test_cnn=np.reshape(X[test],(-1,1,input_dim))
    clf_list = clf_cnn.fit(X_train_cnn, utils.to_categorical(y[train]),epochs=20)   
    y_cnn_probas=clf_cnn.predict(X_test_cnn)
    probas_cnn.append(y_cnn_probas)
    y_class= utils.categorical_probas_to_classes(y_cnn_probas)
    
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]  
    yscore=np.vstack((yscore,y_cnn_probas))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class,y[test])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y[test], y_cnn_probas[:, 1])
    tprs_cnn.append(interp(mean_fpr, fpr, tpr))
    tprs_cnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_cnn.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
print('CNN:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_sum_CNN.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_CNN.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='forestgreen',
lw=lw, label='CNN ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('CNN.csv')