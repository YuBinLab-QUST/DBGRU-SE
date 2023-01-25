import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from imblearn.under_sampling import NearMiss


data_=pd.read_csv(r'zhengfu_GL_0.03.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label#.astype('int64')
nm1 = NearMiss(version=3)
X_resampled, y_resampled = nm1.fit_resample(X, y)
shu2 =X_resampled
shu3 =y_resampled
data_csv = pd.DataFrame(data=shu2)
data_csv.to_csv('A_ADASYN_NM3_GL.csv')
data_csv = pd.DataFrame(data=shu3)
data_csv.to_csv('A_ADASYN_label_NM3_GL.csv')