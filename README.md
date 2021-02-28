# iris
iris basic
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split#to print precision recall etc
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv("/content/drive/My Drive/datasets/Iris.csv")
data.head()

#to check types of data and check if there exist any null values
data.info()

data['Species'].value_counts()

tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species')
plt.show()

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=66)

#knn

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set:',(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set    :',(knn.score(X_test, y_test)))

y_pred=knn.predict(X_test)
print(classification_report(y_test,y_pred))

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 50)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)

print("Training set accuracy:",(lr.score(X_train, y_train)))
print("Test set accuracy:    ",(lr.score(X_test, y_test)))

y_pred=lr.predict(X_test)
print(classification_report(y_test,y_pred))
]
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: ",(tree.score(X_train, y_train)))
print("Accuracy on test set:     ",(tree.score(X_test, y_test)))
prediction = knn.predict([[20, 4.3, 5.5,8]])
y_pred=tree.predict(X_test)
print("Accuracy of the model:",metrics.accuracy_score(y_test, y_pred))

y_pred=tree.predict(X_test)
print(classification_report(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier(n_estimators=100)
rdf.fit(X_train,y_train)
from sklearn import metrics
print("Accuracy on training set: ",(rdf.score(X_train, y_train)))
print("Accuracy on test set:     ",(rdf.score(X_test, y_test)))
y_pred=rdf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test,y_pred))

SepalLength=input('enter SepalLength in Cm=')
SepalWidth=input('enter SepalWidth in Cm=')
PetalLength=input('enter PetalLength in Cm=')
PetalWidth=input('enter PetalWidthCm in Cm=')
answer = rdf.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
print('the flower is =',answer[0])
from sklearn import svm
#------------------ Linear Kernel--------------------
model_linear = svm.SVC(kernel='linear') 
model_linear.fit(X_train, y_train)#Train the model using the training sets
y_pred_linear = model_linear.predict(X_test)#Predict the response for test dataset
#---------------sigmoid kernel-------------
model_sigmoid = svm.SVC(kernel='sigmoid') 
model_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = model_sigmoid.predict(X_test)
#-------------poly kernel-----------
model_poly = svm.SVC(kernel='poly') 
model_poly.fit(X_train, y_train)
y_pred_poly = model_poly.predict(X_test)
#--------------rbf kernel--------------
model_rbf = svm.SVC(kernel='rbf') 
model_rbf.fit(X_train, y_train)
y_pred_rbf = model_rbf.predict(X_test)

#accuracy
acc_linear=np.mean(y_pred_linear==y_test)
acc_sigmoid=np.mean(y_pred_sigmoid==y_test)
acc_poly=np.mean(y_pred_poly==y_test)
acc_rbf=np.mean(y_pred_rbf==y_test)
print('acc using linear as kernel model is ={:.1%}'.format(acc_linear))
print('acc using sigmoid as kernel model is={:.1%}'.format(acc_sigmoid))
print('acc using poly    as kernel model is={:.1%}'.format(acc_poly))
print('acc using rbf     as kernel model is={:.1%}'.format(acc_rbf))
