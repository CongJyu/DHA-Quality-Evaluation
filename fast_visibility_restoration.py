# 从单色或灰度级图像快速恢复可见性 复现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

import cv2
import numpy as np


# 使用白平衡预处理方法
def white_balance(img):
    b, g, r = cv2.split(img)  # 读取图像的三个通道
    b_average = cv2.mean(b)[0]  # 蓝色通道平均值
    g_average = cv2.mean(g)[0]  # 绿色通道平均值
    r_average = cv2.mean(r)[0]  # 红色通道平均值
    k = (r_average + g_average + b_average) / 3
    kb = k / b_average  # 蓝色通道所占增益
    kg = k / g_average  # 绿色通道所占增益
    kr = k / r_average  # 红色通道所占增益
    # 给三种颜色通道增加权重
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    balance_image = cv2.merge([b, g, r])
    return balance_image


# 暗通道
def get_dark_channel(img, size=20):
    # size 是窗口的大小，窗口越大则包含暗通道的概率越大，暗通道就越黑
    r, g, b = cv2.split(img)  # 把图像拆分出 R, G, B 三个通道
    min_channel = cv2.min(r, cv2.min(g, b))  # 取出最小的通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))  # 矩形核
    dark_channel_img = cv2.erode(min_channel, kernel)
    return dark_channel_img


# 去雾
def dehaze(img_input, img_output):
    # TODO: complete dehaze module
    img = cv2.imread(img_input)
    img = img.astype("float32") / 255


# 测试图像读取与白平衡预处理并对比显示
input_path = "./hazed-image/haze-2.jpg"
hazed_image = cv2.imread(input_path)
# dehazed_image = cv2.imread(output_path)
white_balanced_image = white_balance(hazed_image)
cv2.imshow("fast visibility restoration - original", hazed_image)
cv2.imshow("fast visibility restoration - white balanced",
           white_balanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 测试暗通道
dark_channel_image = get_dark_channel(white_balanced_image, size=10)
cv2.imshow("fast visibility restoration - original", hazed_image)
cv2.imshow("fast visibility restoration - dark", dark_channel_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
