# 从单色或灰度级图像快速恢复可见性 复现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

# 计算机视觉及数据处理包
import cv2
import numpy as np
import os
# 质量评价相关函数依赖
import evaluation


# 对图像进行白平衡处理
def white_balance(image):
    '''
    :param image: 读取的图像数据
    :balanced_image: 返回的白平和结果图像
    '''
    red, green, blue = cv2.split(image)
    red_avg = cv2.mean(red)[0]
    green_avg = cv2.mean(green)[0]
    blue_avg = cv2.mean(blue)[0]
    # 每个通道所占增益
    k = (red_avg + green_avg + blue_avg) / 3
    k_red = k / red_avg
    k_green = k / green_avg
    k_blue = k / blue_avg
    red = cv2.addWeighted(
        src1=red,
        alpha=k_red,
        src2=0,
        beta=0,
        gamma=0
    )
    green = cv2.addWeighted(
        src1=green,
        alpha=k_green,
        src2=0,
        beta=0,
        gamma=0
    )
    blue = cv2.addWeighted(
        src1=blue,
        alpha=k_blue,
        src2=0,
        beta=0,
        gamma=0
    )
    balanced_image = cv2.merge(
        [
            blue, green, red
        ]
    )
    return balanced_image


# 获取暗通道
def get_dark_channel(img, size=20):
    # size 是窗口的大小，窗口越大则包含暗通道的概率越大，暗通道就越黑
    r, g, b = cv2.split(img)  # 把图像拆分出 R, G, B 三个通道
    min_channel = cv2.min(r, cv2.min(g, b))  # 取出最小的通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))  # 矩形核
    dark_channel_img = cv2.erode(min_channel, kernel)
    return dark_channel_img


# 推断大气面纱 Atmosphere Veil
# 计算图像 W(x, y) = min(I(x, y))，即获取图像的最小通道
def get_min_channel(image):
    red, green, blue = cv2.split(image)
    min_channel = cv2.min(red, cv2.min(green, blue))
    return min_channel


# 获取大气面纱
def get_atmos_veil(image, sv=41, p=0.95):
    # sv 是中值滤波器的使用的方形窗口的大小
    # 因子 p 在 [0, 1] 范围内，控制可见性恢复的强度
    w_xy = image
    a_xy = cv2.medianBlur(np.uint8(image), sv)
    b_xy = np.abs(w_xy - a_xy)
    b_xy = a_xy - cv2.medianBlur(np.uint8(b_xy), sv)
    max_image = np.ones(b_xy.shape, dtype=np.uint8) * 255
    min_image = cv2.merge(
        [np.uint8(p * 8), np.uint8(w_xy), max_image]
    )
    min_image = get_min_channel(min_image)
    min_image[min_image < 0] = 0
    veil = np.uint8(min_image)
    veil = cv2.blur(veil, (3, 3))
    return veil


# 去雾
def dehaze(img_input, img_output):
    img = cv2.imread(img_input)
    img = img.astype("float32") / 255
    white_balanced_img = white_balance(img)
    atmos_veil = get_atmos_veil(white_balanced_img)
    restored_image = np.zeros(
        (atmos_veil.shape[0], atmos_veil.shape[1], 3),
        dtype=np.float32
    )
    for i in range(0, 3, 1):
        restored_image[:, :, i] = (
            white_balanced_img[:, :, i] - atmos_veil
        ) / (1 - atmos_veil)
    restored_image = restored_image / restored_image.max()
    cv2.imwrite(img_output, restored_image)


# 数据集测试
def dehaze_test(input_path, output_path):
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


if __name__ == "__main__":
    dehaze_test(
        input_path="./hazed-image",
        output_path="./dehazed-image/fast-vis-restoration"
    )
