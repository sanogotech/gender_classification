import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# [height, weight, shoe_size]
X = [[1981],[1988],[1995],[2002],[2007],[2012],[2017]]


Y = [5,7,8,18,22,29,30]

test_data = [[1985],[2027],[3000]]
test_labels = [10,10,10]

#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(X,Y)
dtc_prediction = dtc_clf.predict(test_data)
print(dtc_prediction)

#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(X,Y)
rfc_prediction = rfc_clf.predict(test_data)
print(rfc_prediction)

#Support Vector Classifier
s_clf = SVC()
s_clf.fit(X,Y)
s_prediction = s_clf.predict(test_data)
print(s_prediction)


#LogisticRegression
l_clf = LogisticRegression()
l_clf.fit(X,Y)
l_prediction = l_clf.predict(test_data)
print(l_prediction)

#accuracy scores
dtc_tree_acc = accuracy_score(dtc_prediction,test_labels)
rfc_acc = accuracy_score(rfc_prediction,test_labels)
l_acc = accuracy_score(l_prediction,test_labels)
s_acc = accuracy_score(s_prediction,test_labels)

classifiers = ['Decision Tree', 'Random Forest', 'Logistic Regression' , 'SVC']
accuracy = np.array([dtc_tree_acc, rfc_acc, l_acc, s_acc])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc] + ' is the best classifier for this problem')



