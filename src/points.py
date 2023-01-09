import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def deviation_score(x):
    L = []
    x_mean = np.average(x)
    x_std = np.std(x)
    x_zscore = sp.stats.zscore(X,axis=0)
    count = 0
    for i in x:
        # 標準得点
        # zscore = ((i-x_mean)/x_std)
        zscore = x_zscore[count]
        count = count + 1
        # 偏差値
        L.append(10*zscore+50)
    return L

# 平均arg1点, 標準偏差arg2点, データ数arg3
# X = np.random.normal(50,20,1000)
# Y = np.random.normal(60,10,1000)
var_xx = 400
var_xy = 300
var_yy = 100
XY = np.random.multivariate_normal([50,60], [[var_xx,var_xy],[var_xy,var_yy]], size=500)
X = XY[:,0]
Y = XY[:,1]

# mean 平均
X_mean = np.average(X)
Y_mean = np.average(Y)
print("mean_X    ", X_mean)
print("mean_Y    ", Y_mean)
# variance 分散
X_var = np.var(X)
Y_var = np.var(Y)
print("sigma_X**2", X_var)
print("sigma_Y**2", Y_var)
# standard deviation 標準偏差(分散をもとの単位に戻す)
X_std = np.std(X)
Y_std = np.std(Y)
print("sigma_X    ", X_std)
print("sigma_Y    ", Y_std)
# covariance 共分散行列
cov = np.cov(X,Y)
print("sigma_XX**2", cov[0,0])
print("sigma_XY**2", cov[1,0])
print("sigma_YY**2", cov[1,1])
# 相関係数 -1~1
# 正の相関（一方の値が増えるともう一方の値が増える）
# 負の相関（一方の値が増えるともう一方の値が減る）
coef = np.corrcoef(X,Y)
print("coef_XX    ", coef[0,0])
print("coef_XY    ", coef[1,0])
print("coef_YY    ", coef[1,1])

# 偏差値
# d_score = deviation_score(X)
# print(d_score)

# a = sigma_xy / sigma_x**2
a = cov[0][1] / X_var
# b = mu_y - a*mu_x
b = Y_mean - a*X_mean
# 回帰直線
Y_pred = a * X + b

# 誤差楕円
lambdas, vecs = np.linalg.eigh(cov)
# order = lambdas.argsort()[::-1]
# lambdas, vecs = lambdas[order], vecs[:,order]
n_std = 3
ellip_x = np.sqrt(lambdas[0])*n_std
ellip_y = np.sqrt(lambdas[1])*n_std
ellip_angle_x = np.degrees(np.arctan(vecs[1,0]/vecs[0,0]))
ellip_angle_y = np.degrees(np.arctan(vecs[1,1]/vecs[0,1]))

# figure1
plt.figure(figsize=(10,8),tight_layout=True)
# 1x2, 1st
plt.subplot(221)
labels = ["X", "Y"]
plt.hist([X,Y],ec="black",bins=30,label=labels,alpha=0.5)
plt.legend()
# 1x2, 2nd
plt.subplot(222)
plt.hist(X,ec="black",bins=30,label=labels,alpha=0.5)
plt.hist(Y,ec="black",bins=30,label=labels,alpha=0.5)
plt.legend()
# 2x2, 3rd
plt.subplot(223)
plt.scatter(X, Y)
plt.plot(X, Y_pred)
plt.axis("equal")
# 2x2, 4th
plt.subplot(224)
plt.scatter(X,Y)
plt.plot(X, Y_pred)
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
el = Ellipse(xy=(0,0), width=ellip_x*2, height=ellip_y*2, color="b", alpha=0.3)
transf = transforms.Affine2D().rotate_deg(ellip_angle_x).translate(X_mean, Y_mean)
el.set_transform(transf+plt.subplot(224).transData)
plt.subplot(224).add_patch(el)
plt.axis("equal")

# figure2(object)
fig = plt.figure(figsize=(10,8),tight_layout=True)
# 1x2, 1st
ax1 = fig.add_subplot(221)
labels = ["X", "Y"]
ax1.hist([X,Y],ec="black",bins=30,label=labels,alpha=0.5)
ax1.legend()
# 1x2, 2nd
ax2 = fig.add_subplot(222)
ax2.hist(X,ec="black",bins=30,label=labels,alpha=0.5)
ax2.hist(Y,ec="black",bins=30,label=labels,alpha=0.5)
ax2.legend()
# 2x2, 3rd
ax3 = fig.add_subplot(223)
ax3.scatter(X, Y)
ax3.plot(X, Y_pred)
ax3.axis("equal")
# 2x2, 4th
ax4 = fig.add_subplot(224)
ax4.scatter(X,Y)
ax4.plot(X, Y_pred)
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
el = Ellipse(xy=(0,0), width=ellip_x*2, height=ellip_y*2, color="b", alpha=0.3)
transf = transforms.Affine2D().rotate_deg(ellip_angle_x).translate(X_mean, Y_mean)
el.set_transform(transf+plt.subplot(224).transData)
ax4.add_patch(el)
ax4.axis("equal")

plt.show()