# Lung-Cancer-Detection

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('lung_cancer_examples.csv')
print('Dataset :',data.shape)
data.info()
data[0:10]

data.Result.value_counts()[0:30].plot(kind='bar')
plt.show()

sns.set_style("whitegrid")
sns.pairplot(data,hue="Result",size=3);
plt.show()

data1 = data.drop(columns=['Name','Surname'],axis=1)
data1 = data1.dropna(how='any')
print(data1.shape)

print(data1.shape)
data1.head()

from sklearn.model_selection import train_test_split
Y = data1['Result']
X = data1.drop(columns=['Result'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)

#LogisticRegression
from sklearn.linear_model import LogisticRegression

### Defining the model
logreg = LogisticRegression(C=10)

### We train the model
logreg.fit(X_train, Y_train)

### We predict target values
Y_predict1 = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

logreg_cm = confusion_matrix(Y_test, Y_predict1)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(logreg_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Logistic Regression Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

### Test score
score_logreg = logreg.score(X_test, Y_test)
print(score_logreg)

# Support Vector Machine
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

### We define the SVM model
svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True),n_jobs=-1))

### We train model
svmcla.fit(X_train, Y_train)

### We predict target values
Y_predict2 = svmcla.predict(X_test)


### The confusion matrix
svmcla_cm = confusion_matrix(Y_test, Y_predict2)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

### Test score
score_svmcla = svmcla.score(X_test, Y_test)
print(score_svmcla)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

### We define the model
dtcla = DecisionTreeClassifier(random_state=9)

### We train model
dtcla.fit(X_train, Y_train)

### We predict target values
Y_predict4 = dtcla.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
dtcla_cm = confusion_matrix(Y_test, Y_predict4)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(dtcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Decision Tree Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

### Test score
score_dtcla = dtcla.score(X_test, Y_test)
print(score_dtcla)

Testscores = pd.Series([score_logreg, score_svmcla, score_dtcla], 
                        index=['Logistic Regression Score', 'Support Vector Machine Score', 'Decision Tree Score']) 
print(Testscores)
# 
ROC Curve
from sklearn.metrics import roc_curve

### Logistic Regression Classification
Y_predict1_proba = logreg.predict_proba(X_test)
Y_predict1_proba = Y_predict1_proba[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_predict1_proba)
plt.subplot(331)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve Logistic Regression')
plt.grid(True)

### SVM Classification
Y_predict2_proba = svmcla.predict_proba(X_test)
Y_predict2_proba = Y_predict2_proba[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_predict2_proba)
plt.subplot(332)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve SVM')
plt.grid(True)

### Decision Tree Classification
Y_predict4_proba = dtcla.predict_proba(X_test)
Y_predict4_proba = Y_predict4_proba[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_predict4_proba)
plt.subplot(334)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve Decision Tree')
plt.grid(True)


