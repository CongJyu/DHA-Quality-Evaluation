# 暗通道先验去雾

import cv2
import numpy as np


# 计算含雾图像的暗通道
def dark_channel(img, size=15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))  # 取出最暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img


# 估计全局大气光值
def get_atom(img, percent=0.001):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)


# 估算投射率
def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t


# 引导滤波
def guided_filter(p, i, r, e):
    # p: input image
    # i: guidance image
    # r: radius
    # e: regularization
    # return: filtering output q

    # 1
    mean_i = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_i = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    # 3
    a = cov_ip / (var_i + e)
    b = mean_p - a * mean_i
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5
    q = mean_a * i + mean_b

    return q


# 去雾
def dehaze(im):
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atom = get_atom(img)
    trans = get_trans(img, atom)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    return result * 255
