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
import seaborn
from sklearn.cross_validation import train_test_split as tts
from sklearn.cross_validation import cross_val_score
%matplotlib inline

df=pd.read_csv('titanic.csv')

#The assumptions of this data is that it is accurate and represents the entire population of the titanic.
#There are less than 900 individuals listed in this data and there were almost 3000 onboard the titanic.
#Therefor, we are taking a risk in assuming that this population is repesentative
#and we are also taking a risk in assuming that these old records are accurate.
df.head()

df.isnull().sum()
df.Parch.unique()

df=df.drop(['Cabin','Ticket','Name','index','PassengerId'], axis=1)
df=df.dropna()

df.describe()

df.dtypes
df.Age.sort_values()

df=pd.get_dummies(df)
df.dtypes

y=df.Survived
x=df.drop('Survived',axis=1)

#makes a baseline for accuracy
float(len(y[y==0]))/float(len(y))

model=lr()
model.fit(x,y)
model.score(x,y)

for column in x.columns:
    print column, np.corrcoef(x[column],y)[1][0]

x_train, x_test, y_train, y_test = tts(x,y, train_size=.8, random_state=1)

#train/test fitting and validation
model.score(x_test,y_test)
proba=model.predict_proba(x_test)
pred=model.predict(x_test)

s= cross_val_score(model,x_test,y_test, cv=5)
s.mean()
s.std()

#The f-1 scores show that our model does a fairly decent job of predicting those
#who died and an okay job predicting those who survived
print skcr(y_test,pred)


#(true negative) (false Negative)
#(false negative) (true positive)
print skcm(y_test,pred)

#Cant remember the better way to parse this
probs=[]
for item in proba:
    probs.append(item[1])

false_positive_rate, true_positive_rate, thresholds = skrc(y_test,probs)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'r', label=roc_auc)
plt.legend(loc='lower right')
plt.show()
