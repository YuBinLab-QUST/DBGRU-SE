import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from imblearn.combine import SMOTEENN


data_=pd.read_csv(r'NR_Group_Lasso_0.03.csv')
data1=np.array(data_)
data=data1[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label#.astype('int64')
X_resampled, y_resampled = SMOTEENN(sampling_strategy={1:81000}).fit_resample(X, y)
shu2 =X_resampled
shu3 =y_resampled
data_csv = pd.DataFrame(data=shu2)
data_csv.to_csv('SMOTEENN_GL_0.03_YZ.csv')
data_csv = pd.DataFrame(data=shu3)
data_csv.to_csv('SMOTEENN_label_GL_0.03_YZ.csv')