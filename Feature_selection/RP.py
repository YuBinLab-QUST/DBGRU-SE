import numpy as np
from sklearn import random_projection
from sklearn.preprocessing import scale
import pandas as pd



data_train=pd.read_csv(r'zhengfu.csv')
#yeast_data=sio.loadmat('DNN_yeast_six.mat')
data_=np.array(data_train)
data=data_[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
rp = random_projection.SparseRandomProjection(n_components=200, random_state=66)
X_projected = rp.fit_transform(shu)
shu=X_projected
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('Random projection.csv')