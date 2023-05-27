import numpy as np
import cv2

img = 'lenna.png'
img = cv2.imread(img)


def nearest(imgs, H=800, W=800):
    h, w, c = imgs.shape
    img_blank = np.zeros((H, W, c), np.uint8)
    h_ = H / h
    w_ = W / w
    for i in range(H):
        for j in range(W):
            x = int(i / h_ + 0.5)
            y = int(j / w_ + 0.5)
            img_blank[i, j] = imgs[x, y]
    return img_blank


img1 = nearest(img)
cv2.imshow('1', img1)
cv2.imshow('0', img)
cv2.waitKey()
