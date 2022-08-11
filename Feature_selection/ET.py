import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from dimensional_reduction import selectFromExtraTrees


data_=pd.read_csv(r'zhengfu.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)	
data_2,importance=selectFromExtraTrees(shu,label)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('ET.csv')