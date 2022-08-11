import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from dimensional_reduction import Light_lasso

data_=pd.read_csv(r'zhengfu.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))#Value can be changed
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label
data_2,importance=Light_lasso(X,y,0.03)#改参数
shu=data_2 
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('Group_Lasso_0.03.csv')
data_csv = pd.DataFrame(data=importance)
data_csv.to_csv('GL_importance_0.03.csv')