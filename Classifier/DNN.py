import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils
from sklearn.preprocessing import scale,StandardScaler
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten


def get_shuffle(dataset,label):    
    #shuffle data
    #np.random.seed(random_state)
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label  

data_=pd.read_csv(r'SMOTEENN_GL_0.03.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X,y=get_shuffle(shu,label)
[m,n]=np.shape(X)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

#

def get_DNN_model():
    input_1 = Input(shape=(n,), name='drug')
    drug_input1 = Dense(int(n), activation='relu', init='glorot_normal', name='High_dim_feature_1')(input_1)
    drug_input1=Dropout(0.5)(drug_input1)
    drug_input1 = Dense(int(n/2), activation='relu', init='glorot_normal', name='High_dim_feature_2')(drug_input1)
    drug_input1=Dropout(0.5)(drug_input1)
    drug_input1 = Dense(int(n/4), activation='relu', init='glorot_normal', name='High_dim_feature_3')(drug_input1)
    drug_input1=Dropout(0.5)(drug_input1)
    output = Dense(int(n/8), activation='relu', init='glorot_normal', name='High_dim_feature')(drug_input1)
    outputs = Dense(2, activation='softmax', name='output')(output)
    model = Model(input=input_1, output=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])#generate the resonable results
    cv_clf =get_DNN_model()
    hist=cv_clf.fit(X[train], 
                    y_train,
                    epochs=20)
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]       
    y_score=cv_clf.predict(X[test])#the output of  probability
    yscore=np.vstack((yscore,y_score))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    
    print('DNN:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
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
yscore_sum.to_csv('yscore_sum_DNN.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_DNN.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='deeppink',
lw=lw, label='DNN ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp','MCC','F1','AUC']
ro=['1', '2', '3','4','5','6','7','8','9','10','11']
data_csv = pd.DataFrame(columns=colum, data=result,index=ro)
data_csv.to_csv(r'DNN.csv')