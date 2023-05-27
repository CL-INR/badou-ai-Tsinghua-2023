import cv2
import numpy as np
import random as R
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


image = cv2.imread('lenna.png')
img = image.copy()
img1 = image.copy()
# cv2.imshow('img', img)
w, h, c = img.shape

# ————————————————————高斯分布散点数据————————————————————————
def Gaussian_Distribution(N=2, M=1000, m=256, sigma=4900):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差
    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma
    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    return data, Gaussian
# 散点图
data, _ = Gaussian_Distribution()
x, y = data.T
for i in range(len(x)):
    x[i] = int(x[i])
for i in range(len(y)):
    y[i] = int(y[i])
print(data.T)
plt.scatter(x, y)
plt.show()

# ————————————————————高斯噪声——————————————————————————————
def NoiseG(img):
    for acc in range(c):
        for i in range(h):
            for j in range(w):
                if i in x and j in y:
                    img[i][j][acc] = img[i][j][acc] + R.gauss(4, 2)
                if img[i][j][acc] > 255:
                    img[i][j][acc] = 255
    return img

# —————————————————————————椒盐噪声————————————————————————————
num_img = w * h
def Random(n):
    x = []
    for i in range(int(0.1 * n)):
        x.append(R.randint(0, n))
    return x
def NoiseJ(img1):
    X = Random(num_img)
    Y = Random(num_img)
    for i in range(h):
        for j in range(w):
            a = R.randint(0, 1)
            if i in X and j in Y and a == 1:
                img1[i][j] = [255, 255, 255]
            elif i in X and j in Y and a == 0:
                img1[i][j] = [0, 0, 0]
    return img1


img = NoiseG(img)
img1 = NoiseJ(img1)
cv2.imshow('noise_G', img)
cv2.imshow('noise_J', img1)
cv2.imshow('original', image)
cv2.waitKey()
