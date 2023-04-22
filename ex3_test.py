from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# 读取数据
data = pd.read_csv('train3.txt', sep=' ', header=1)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义SVM分类器
C1=1000
C2=100
clf = SVC(C=C1,kernel='rbf')
clf1 = SVC(C=C1,kernel='rbf',gamma=0.1)
clf2 = SVC(C=C1,kernel='rbf',gamma=1)
clf3 = SVC(C=C1,kernel='rbf',gamma=10)
clf4 = SVC(C=C1,kernel='rbf',gamma=100)
clf5 = SVC(C=C2,kernel='rbf')
clf6 = SVC(C=C2,kernel='rbf',gamma=0.1)
clf7 = SVC(C=C2,kernel='rbf',gamma=1)
clf8 = SVC(C=C2,kernel='rbf',gamma=10)
clf9 = SVC(C=C2,kernel='rbf',gamma=100)

# 训练SVM分类器
clf.fit(X_train, y_train)
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)
clf5.fit(X_train, y_train)
clf6.fit(X_train, y_train)
clf7.fit(X_train, y_train)
clf8.fit(X_train, y_train)
clf9.fit(X_train, y_train)

# 使用训练好的分类器对测试集进行预测
y_pred = clf.predict(X_test)
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf4.predict(X_test)
y_pred5 = clf5.predict(X_test)
y_pred6 = clf6.predict(X_test)
y_pred7 = clf7.predict(X_test)
y_pred8 = clf8.predict(X_test)
y_pred9 = clf9.predict(X_test)

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy3 = accuracy_score(y_test, y_pred3)
accuracy4 = accuracy_score(y_test, y_pred4)
accuracy5 = accuracy_score(y_test, y_pred5)
accuracy6 = accuracy_score(y_test, y_pred6)
accuracy7 = accuracy_score(y_test, y_pred7)
accuracy8 = accuracy_score(y_test, y_pred8)
accuracy9 = accuracy_score(y_test, y_pred9)


print('Accuracy:{} C=1000 gamma={}'.format(accuracy,clf.gamma))
print('Accuracy1:{} C=1000 gamma={}'.format(accuracy1,0.1))
print('Accuracy2:{} C=1000 gamma={}'.format(accuracy2,1))
print('Accuracy3:{} C=1000 gamma={}'.format(accuracy3,10))
print('Accuracy4:{} C=1000 gamma={}'.format(accuracy4,100))
print('Accuracy5:{} C=100 gamma={}'.format(accuracy5,clf5.gamma))
print('Accuracy6:{} C=100 gamma={}'.format(accuracy6,0.1))
print('Accuracy7:{} C=100 gamma={}'.format(accuracy7,1))
print('Accuracy8:{} C=100 gamma={}'.format(accuracy8,10))
print('Accuracy9:{} C=100 gamma={}'.format(accuracy9,100))