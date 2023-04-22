from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')
    ax.grid(axis='x', linestyle='-.')


def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1
    neg = y_ == -1
    ax = plt.subplot()

    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(0, 0.8, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    Z_rbf = clf.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

    # ax.contourf(X_, Y_, Z_rbf, alpha=0.75)
    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})

    set_ax_gray(ax)

    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    support_labels = y_[clf.support_]
    support_colors = np.where(support_labels == 1, 'c','lightcoral')
    ax.scatter(X_[clf.support_, 0], X_[clf.support_, 1], marker='o', c=support_colors, edgecolors='g', s=150,
               label='support_vectors')

    ax.legend()
    ax.set_title('{} kernel, C={}'.format(kernel, c))
    plt.show()

df = pd.read_excel('watermelon.xlsx', )
df.to_csv('train3.txt', index=False,sep=' ')
path = r'E:\Python-Try\machinelearning\example3\train3.txt'
data = pd.read_table(path, delimiter=' ', dtype=float)

X = data.iloc[:, [0, 1]].values
y = data.iloc[:, 2].values

y[y == 0] = -1

C = 1000

clf_rbf = svm.SVC(C=C)
clf_rbf1 = svm.SVC(C=C,gamma=0.1)
clf_rbf2 = svm.SVC(C=C,gamma=1)
clf_rbf3 = svm.SVC(C=C,gamma=10)
clf_rbf4 = svm.SVC(C=C,gamma=100)
# bandwidth = clf_rbf.gamma if clf_rbf.gamma != 'scale' else 1.0 / (X.shape[1] * X.var())
# print(clf_rbf.gamma)
# print(1.0 / (X.shape[1] * X.var()))
clf_rbf.fit(X, y.astype(int))
clf_rbf1.fit(X, y.astype(int))
clf_rbf2.fit(X, y.astype(int))
clf_rbf3.fit(X, y.astype(int))
clf_rbf4.fit(X, y.astype(int))
print('高斯核：')
print('gamma=',clf_rbf.gamma)
print('预测值：', clf_rbf.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf.support_)
print(' ')
print('gamma=',clf_rbf1.gamma)
print('预测值：', clf_rbf1.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf1.support_)
print(' ')
print('gamma=',clf_rbf2.gamma)
print('预测值：', clf_rbf2.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf2.support_)
print(' ')
print('gamma=',clf_rbf3.gamma)
print('预测值：', clf_rbf3.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf3.support_)
print(' ')
print('gamma=',clf_rbf4.gamma)
print('预测值：', clf_rbf4.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_rbf4.support_)

print('-' * 40)
clf_linear = svm.SVC(C=C, kernel='linear')
clf_linear.fit(X, y.astype(int))
print('线性核：')
print('预测值：', clf_linear.predict(X))
print('真实值：', y.astype(int))
print('支持向量：', clf_linear.support_)

plt_support_(clf_rbf, X, y, 'rbf', C)
plt_support_(clf_rbf1, X, y, 'rbf', C)
plt_support_(clf_rbf2, X, y, 'rbf', C)
plt_support_(clf_rbf3, X, y, 'rbf', C)
plt_support_(clf_rbf4, X, y, 'rbf', C)

plt_support_(clf_linear, X, y, 'linear', C)
