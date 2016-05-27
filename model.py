import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression as lr
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report as skcr
from sklearn.metrics import confusion_matrix as skcm
from sklearn.metrics import roc_curve as skrc
from sklearn.metrics import auc
from sklearn.grid_search import GridSearchCV as skgs
import seaborn
from sklearn.cross_validation import train_test_split as tts
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
%matplotlib inline

df=pd.read_csv('titanic.csv')

#The assumptions of this data is that it is accurate and represents the entire population of the titanic.
#There are less than 900 individuals listed in this data and there were almost 3000 onboard the titanic.
#Therefor, we are taking a risk in assuming that this population is repesentative
#and we are also taking a risk in assuming that these old records are accurate.
df.head()

#looking at data and cleaning table
df.isnull().sum()
df.Parch.unique()
df=df.drop(['Cabin','Ticket','Name','index','PassengerId'], axis=1)
df=df.dropna()
df.describe()
df.dtypes
df.Age.sort_values()

#getting dummies for categories
df=pd.get_dummies(df)
df.dtypes

#creating targets and data set
y=df.Survived
x=df.drop('Survived',axis=1)

#makes a baseline for accuracy
float(len(y[y==0]))/float(len(y))

#basic model, no validation
model=lr()
model.fit(x,y)
model.score(x,y)

#looking at correlations
for column in x.columns:
    print column, np.corrcoef(x[column],y)[1][0]

#building test train sets
x_train, x_test, y_train, y_test = tts(x,y, train_size=.8, random_state=1)

#train/test fitting and validation
model.fit(x_train,y_train)
model.score(x_test,y_test)
proba=model.predict_proba(x_test)
pred=model.predict(x_test)
s= cross_val_score(model,x_test,y_test, cv=12)
s.mean()
s.std()

#The f-1 scores show that our model does a fairly decent job of predicting those
#who died and an okay job predicting those who survived
print skcr(y_test,pred)

#(true negative) (false positive)
#(false negative) (true positive)
print skcm(y_test,pred)

#Cant remember the better way to parse through this
probs=[]
for item in proba:
    probs.append(item[1])

#building ROC
false_positive_rate, true_positive_rate, thresholds = skrc(y_test,probs)
roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting ROC
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'r', label=roc_auc)
plt.legend(loc='lower right')
plt.show()

#Setting up gridsearch
logreg_parameters = {'penalty':['l1','l2'],'C':np.logspace(-5,1,50),'solver':['liblinear']}
grid=skgs(model,logreg_parameters,cv=12,scoring='accuracy')
grid.fit(x_train,y_train)
print grid.best_estimator_
print grid.score(x_test,y_test)

#L1 (lasso) penalties will reduce certian coefficients in a way that pushes features
#influence towards zero, effectively eliminating the weight of some features
#L2 (Ridge) penalities will generally rein in all coefficients so that they
#do not overfit the model
#ridge is helpful if we are afriad of making our model too bias due to our training data
#lasso is helpful if we are unsure of which features we should try to eliminate

#C is related to the amount of penalty we apply. C is the inverse of alpha. Alpha
#is the multiplier we use to apply penalty. As alpha increases, more penalty is
#added, and the model becomes more sensitive to larger coefficients

#If we change our threshold to .90, we would be much more confident that our
#predicted survivals were true survivals. However, we would become less confident
#that our deaths were true death. We would reduce our false positive rate, but
#increase our false negative rate.


#KNN
#does not perform as well as LogReg, even with GridSeach
knn =knn()
knn.fit(x_train,y_train)
knn.score(x_test,y_test)

gridknn=skgs(knn,{'n_neighbors':range(1,55)},cv=12,scoring='accuracy')
gridknn.fit(x_train,y_train)
print gridknn.best_estimator_
print gridknn.score(x_test,y_test)
#As we use more neighbors, our model becomes more bias because it becomes more complex
#Logistice regression is usually a better choice than KNN because it is a more sophisticated
#model and it requires less storage to run. KNN is a good model if you are looking
#for something simple, the data set is not too large, and you want as transparent
#of a model as possible.
knnpred=gridknn.predict(x_test)
print skcm(y_test,knnpred)
#with knn, we got more true negatives, and less false positives,
#but we did much worse at classifying true positives from false negatives.



#ROC for both tests
probaknn=gridknn.predict_proba(x_test)
probsknn=[]
for item in probaknn:
    probsknn.append(item[1])


false_positive_rateknn, true_positive_rateknn, thresholdsknn = skrc(y_test,probsknn)
roc_aucknn = auc(false_positive_rateknn, true_positive_rateknn)


plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'r', label='LogReg %f'%roc_auc)
plt.plot(false_positive_rateknn, true_positive_rateknn, 'b', label='KNN %f'%roc_aucknn)
plt.legend(loc='lower right')
plt.show()

#LogReg Gridsearch with average_percision
gridap=skgs(model,logreg_parameters,cv=12,scoring='average_precision')
gridap.fit(x_train,y_train)
print gridap.best_estimator_
gridap.score(x_test,y_test)
#this model scores much better than the logreg accuracy model in part 5

appred=gridap.predict(x_test)
print skcm(y_test,appred)
#The confision matrix is almost the same as regular logreg, but with one more false
#positive, which is strange since this model actually scored better

probaap=gridap.predict_proba(x_test)
probsap=[]
for item in probaap:
    probsap.append(item[1])

probsap

len(y_test)
len(probsap)
false_positive_rateap, true_positive_rateap, thresholdsap = skrc(y_test,probsap)
roc_aucap = auc(false_positive_rateap, true_positive_rateap)

plt.title('ROC')
plt.plot(false_positive_rateap, true_positive_rateap, 'b', label='LogReg (AP) %f'%roc_aucap)
plt.legend(loc='lower right')
plt.show()
