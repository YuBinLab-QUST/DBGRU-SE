import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.preprocessing import normalize,Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.linear_model import LassoLarsCV,LassoLars
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from lightning.classification import CDClassifier 

##using selectFromExtraTrees to reduce the dimension
def selectFromExtraTrees(data,label):
    clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, 
                              class_weight=None)#entropy )#entropy
    clf.fit(data,label)
    importance=clf.feature_importances_ 
    model=SelectFromModel(clf,prefit=True)
    new_data = model.transform(data)
    return new_data,importance
 
def Light_lasso(X,y,alpha_):
     clf = CDClassifier(penalty="l1/l2",                    
                   loss="squared_hinge",                    
                   #multiclass=True,                    
                   max_iter=50,                    
                   alpha=alpha_,                    
                   C=1.0 / X.shape[0],                    
                   tol=1e-3)  
     clf.fit(X, y  ) 
     H1,H2=np.nonzero(clf.coef_)
     X=X[:,H2]
     return X,H2
     
