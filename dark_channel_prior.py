# 暗通道先验去雾 复现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

import cv2
import numpy as np
import os
import evaluation


# 根据 Kaiming He 的论文描述求取图像的暗通道
def get_dark_channel(img, size=20):
    # size 是窗口的大小，窗口越大则包含暗通道的概率越大，暗通道就越黑
    r, g, b = cv2.split(img)  # 把图像拆分出 R, G, B 三个通道
    min_channel = cv2.min(r, cv2.min(g, b))  # 取出最小的通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))  # 矩形核
    dark_channel_img = cv2.erode(min_channel, kernel)
    return dark_channel_img

# 去雾模型 I(x) = J(x)t(x) + A[1 - t(x)]
# 常量参数 w (0 < w < 1) 控制去雾的程度
# 透射率 t(x) = 1 - w * min(I_c(y) / A_c)
# J(x) = (I(x) - A) / (max(t(x), t_0)) + A


# 估计全球大气光
# percent 指定图像中最亮像素的百分比
def get_atmos_light(img, percent=0.001):
    mean_per_pixel = np.mean(img, axis=2).reshape(-1)  # 计算图像每个像素的平均值，拉平成一维数组
    # 选取最亮 percent 部分的像素
    mean_per_pixel = np.sort(mean_per_pixel)[::-1]
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


# 数据集评估
def dehaze_evaluate(input_path, output_path):
    input_path_list = os.listdir(input_path)
    if ".DS_Store" in input_path_list:
        input_path_list.remove(".DS_Store")
    elif input_path_list is not None:
        print("[ FAIL ] Original image path is empty.")
    output_path_list = os.listdir(output_path)
    if ".DS_Store" in output_path_list:
        output_path_list.remove(".DS_Store")
    elif output_path_list is not None:
        print("[ FAIL ] Dehazed image path is empty.")

    print("Image    \tPSNR     \tSSIM\n---------\t---------\t---------")
    for file_name in input_path_list:
        # original_image_path = os.path.join(input_path, file_name)
        original_image_path = os.path.join("./clear-image", file_name)
        dehazed_image_path = os.path.join(output_path, file_name)
        original_image = cv2.imread(original_image_path)
        original_image = original_image.astype("float32") / 255
        dehazed_image = cv2.imread(dehazed_image_path)
        dehazed_image = dehazed_image.astype("float32") / 255
        current_psnr = round(evaluation.compare_psnr(
            original_image, dehazed_image), 6)
        current_ssim = round(evaluation.compare_ssim(
            original_image, dehazed_image, win_size=7, data_range=255, channel_axis=2), 6)
        print(file_name, "\t", current_psnr, "\t", current_ssim)


# 测试图像读取并对比显示
# hazed_image = cv2.imread(input_path)
# dehazed_image = cv2.imread(output_path)
# dehaze(hazed_image, dehazed_image)
# cv2.imshow("dark channel prior dehazing - original", hazed_image)
# cv2.imshow("dark channel prior dehazing - processed", dehazed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def get_dark_channel_image(input_path, output_path):
    input_path_list = os.listdir(input_path)
    if ".DS_Store" in input_path_list:
        input_path_list.remove(".DS_Store")
    output_path_list = os.listdir(output_path)
    if ".DS_Store" in output_path_list:
        output_path_list.remove(".DS_Store")

    for file_name in input_path_list:
        input_hazed_image = os.path.join(input_path, file_name)
        output_dcp_image = os.path.join(output_path, file_name)
        img = cv2.imread(input_hazed_image)
        print("[ INFO ] Generating dark channel prior image: ", file_name)
        single_dcp_channel = get_dark_channel(img=img, size=15)
        dcp_img = cv2.merge(
            [single_dcp_channel, single_dcp_channel, single_dcp_channel]
        )
        cv2.imwrite(output_dcp_image, dcp_img)


if __name__ == "__main__":
    dehaze_test(
        input_path="./hazed-image",
        output_path="./dehazed-image/dark-channel-prior"
    )
    dehaze_evaluate(
        input_path="./hazed-image",
        output_path="./dehazed-image/dark-channel-prior"
    )
