# 基于 Tarel 提出的中值滤波的图像去雾方法

# 基本方法是基于中值滤波估计尘雾浓度，利用大气散射模型恢复无雾的图像
# 使用白平衡方法对含雾图像进行预处理

import PIL.Image
import PIL.ImageFilter
import cv2
import numpy as np
import PIL
import os

# 处理白平衡
def white_balance(image):
    r, g, b = cv2.split(image)  # 读取图像的三个通道
    r_average = cv2.mean(r)[0]  # 红色通道平均值
    g_average = cv2.mean(g)[0]  # 绿色通道平均值
    b_average = cv2.mean(b)[0]  # 蓝色通道平均值
    k = (r_average + g_average + b_average) / 3
    kr = k / r_average  # 红色通道所占增益
    kg = k / g_average  # 绿色通道所占增益
    kb = k / b_average  # 蓝色通道所占增益
    # 给三种颜色通道增加权重
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_image = cv2.merge([b, g, r])
    return balance_image


# 处理暗通道
def get_dark_channel(image, block_size, rgb_atoms):
    image_gray = np.float64(image)
    min_image = get_min_channel(image, rgb_atoms)
    image = PIL.Image.fromarray(min_image)
    image = image.filter(PIL.ImageFilter.MinFilter(block_size))
    dark_channel = np.asarray(image, dtype=np.float64)
    return dark_channel


# 获取最小灰度图
def get_min_channel(image):
    image_gray = np.zeros((image.shape[0], image.shape[1]), np.float32)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            local_min = 255
            for k in range(0, 2):
                if image.item((i, j, k)) < local_min:
                    local_min = image.item((i, j, k))
            image_gray[i, j] = local_min
    return image_gray


# 窗口显示
def cv_display(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 测试图像位置
test_image_path = r'/Users/rainchen/Coding/DHA-Quality-Evaluation/sample-image/single_image'
file_list = os.listdir(test_image_path)

s_v = 41  # 中值滤波器的尺寸大小
p = 0.95  # 天空光比例的大小

for file_name in file_list:
    original_image = cv2.imread(os.path.join(test_image_path, file_name))
    original_image = cv2.resize(original_image, dsize=None, fx=0.2, fy=0.2)
    original_white_balance = white_balance(original_image)

    W = get_min_channel(original_white_balance)

    A = cv2.medianBlur(np.uint8(W), s_v)
    B = np.abs(W - A)
    B = A - cv2.medianBlur(np.uint8(B), s_v)
    max_255_image = np.ones(B.shape, dtype=np.uint8) * 255
    min_t = cv2.merge([np.uint8(p * B), ])
    min_t = get_min_channel(min_t)
    min_t[min_t < 0] = 0
    V = np.uint8(min_t)
    V = cv2.blur(V, (5, 5))  # 平滑滤波

    cv2.imwrite('test_dehaze/V_' + file_name, V)
    V = np.float32(V) / 255
    R_dehazy = np.zeros((V.shape[0], W.shape[1], 3), dtype=np.float32)
    original_white_balance = np.float32(original_white_balance) / 255

    for i in range(0, 3, 1):
        R_dehazy[:, :, i] = (original_white_balance[:, :, i] - V) / (1 - V)
    R_dehazy = R_dehazy / R_dehazy.max()

    R_dehazy = np.clip(R_dehazy, 0, 1)
    R_dehazy = np.uint8(R_dehazy * 255)

    src = original_image
    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0:h, 0:w, :] = original_image
    result[0:h, w:2 * w, :] = R_dehazy
    cv2.putText(result, "orignal_image", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)
    cv2.putText(result, "dehazed_image", (w + 10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)

    cv2.imwrite('result_image' + file_name, result)    
