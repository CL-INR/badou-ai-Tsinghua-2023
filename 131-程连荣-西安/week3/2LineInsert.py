import numpy as np
import cv2


def Insert2L(img, out_dim):
    org_h, org_w, channel = img.shape
    out_h, out_w = out_dim[1], out_dim[0]
    print("org_h, org_w = ", org_h, org_w)
    print("out_h, out_w = ", out_h, out_w)
    if org_h == out_h and org_w == out_w:
        return img.copy()
    img1 = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(org_w) / out_w, float(org_h) / out_h
    for i in range(3):
        for y in range(out_h):
            for x in range(out_w):
                o_x = (x + 0.5) * scale_x - 0.5
                o_y = (y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(o_x))
                src_x1 = min(src_x0 + 1, org_w - 1)
                src_y0 = int(np.floor(o_y))
                src_y1 = min(src_y0 + 1, org_h - 1)
                # 分母为1，(x2-x)f(Q11)+(x-x1)f(Q21)
                temp0 = (src_x1 - o_x) * img[src_y0, src_x0, i] + (o_x - src_x0) * img[src_y0, src_x1, i]
                # (x2-x)f(Q12)+(x-x1)f(Q12)
                temp1 = (src_x1 - o_x) * img[src_y1, src_x0, i] + (o_x - src_x0) * img[src_y1, src_x1, i]
                # (y2-y)f(R1)+(y-y1)f(R2)
                img1[y, x, i] = int((src_y1 - o_y) * temp0 + (o_y - src_y0) * temp1)

    return img1


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img1 = Insert2L(img, (800, 800))
    cv2.imshow('2LineInsert', img1)
    cv2.waitKey()
