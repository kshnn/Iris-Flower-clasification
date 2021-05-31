#Data collection
import pandas as pd
data=pd.read_csv('iris.csv')

#Data clearing
data.drop('Id',axis=1,inplace=True)

#Replacing target objects
Classes={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
data.replace({"Species":Classes},inplace=True)


#Create arrays [/xFeatures] & [y/target]
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#Data Visualization (optional)
'''pairplot
Library:seaborn'''
# import seaborn as sb
# sb.pairplot(data)

#Data Splitting
'''
Library: sklearn
M1     : model_selection
M2     : train_test_split
'''
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=30)

#Algorithm Selection

'''
Library: sklearn
M1     : linear_model
Class  : LogisticRegression

learn  : fit()
predict: predict()
accu.  : score()
'''

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(xtrain,ytrain)
logreg_Ypedct=logreg.predict(xtest)
logrec_Acc=logreg.score(xtest,ytest)

#Confusion Matrix
'''
Library: sklearn
Module : metrics
fn  : confusion_matrix(actual,predicted)
'''
from sklearn.metrics import confusion_matrix as confuMat
conmat=confuMat(ytest,logreg_Ypedct)

#Saving model
'''
using pickle:
    save model-dump(model name, file object)
    open model-load()
    
'''


filename='log_reg_iris.sav'
file_is_ready=open(filename, 'wb')

import pickle
pickle.dump(logreg,file_is_ready)

'''
using joblib:
    dump(model name,file name)
    load(file name)
'''

import joblib
joblib.dump(logreg,'logistic_iris.pkl')

my_loaded_model=joblib.load('logistic_iris.pkl')






