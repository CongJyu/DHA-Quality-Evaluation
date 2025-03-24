# 从单色或灰度级图像快速恢复可见性 复现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

import cv2
import numpy as np
import os


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


# 根据 Kaiming He 的操作步骤
# 估计全球大气光
# percent 指定图像中最亮像素的百分比
def get_atmos_light(img, percent=0.001):
    mean_per_pixel = np.mean(img, axis=2).reshape(-1)  # 计算图像每个像素的平均值，拉平成一维数组
    # 选取最亮 percent 部分的像素
    mean_top = mean_per_pixel[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_top)


# 估计透射率
def get_trans(img, atm, w=0.95):
    x = img / atm
    t = 1 - w * get_dark_channel(x, 20)
    return t


# 引导滤波
def guided_filter(p, i, r, e):
    # p: input image
    # i: guidance image
    # r: radius
    # e: regularization
    # return: filtering output q

    # 1
    mean_i = cv2.boxFilter(i, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    corr_i = cv2.boxFilter(i * i, cv2.CV_32F, (r, r))
    corr_ip = cv2.boxFilter(i * p, cv2.CV_32F, (r, r))
    # 2
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    # 3
    a = cov_ip / (var_i + e)
    b = mean_p - a * mean_i
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    # 5
    q = mean_a * i + mean_b

    return q


# 去雾
def dehaze(img_input, img_output):
    img = cv2.imread(img_input)
    img = img.astype("float32") / 255
    img = white_balance(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32") / 255
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    atmos_light = get_atmos_light(img)
    trans = get_trans(img, atmos_light)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atmos_light) / \
            trans_guided + atmos_light
    cv2.imwrite(img_output, result * 255)


# 数据集测试
def test(input_path, output_path):
    input_path_list = os.listdir(input_path)
    if ".DS_Store" in input_path_list:
        input_path_list.remove(".DS_Store")
    output_path_list = os.listdir(output_path)
    if ".DS_Store" in output_path_list:
        output_path_list.remove(".DS_Store")

    for file_name in input_path_list:
        input_hazed_image = os.path.join(input_path, file_name)
        output_dehazed_image = os.path.join(output_path, file_name)
        print("[ INFO ] Processing image: ", file_name)
        dehaze(input_hazed_image, output_dehazed_image)


# 测试图像读取与白平衡预处理并对比显示
# input_path = "./hazed-image/haze-2.jpg"
# output_white_balanced_path = "./dehazed-image/white-balanced-2.jpg"
# output_dark_channel_path = "./dehazed-image/dark-channel-2.jpg"
# output_dehazed_path = "./dehazed-image/dehazed-2.jpg"
# hazed_image = cv2.imread(input_path)
# white_balanced_image = white_balance(hazed_image)
# cv2.imwrite(output_white_balanced_path, white_balanced_image)
# cv2.imshow("fast visibility restoration - original", hazed_image)
# cv2.imshow("fast visibility restoration - white balanced",
#            white_balanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 测试暗通道
# dark_channel_image = get_dark_channel(white_balanced_image, size=10)
# cv2.imwrite(output_dark_channel_path, dark_channel_image)
# cv2.imshow("fast visibility restoration - original", hazed_image)
# cv2.imshow("fast visibility restoration - dark", dark_channel_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 测试去雾效果
# dehaze(output_white_balanced_path, output_dehazed_path)
# dehazed_image = cv2.imread(output_dehazed_path)
# cv2.imshow("fast visibility restoration - original", hazed_image)
# cv2.imshow("fast visibility restoration - dehazed", dehazed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
