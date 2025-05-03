# 直方图均衡化去雾 复现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

# 计算机视觉及数据处理包
import cv2
import numpy as np
import os
# 质量评价相关函数依赖
import evaluation


# 传统图像增强方法
def dehaze(img_input, img_output):
    # 读取含雾图像（彩色）
    img = cv2.imread(img_input)
    # 拆分颜色通道
    b, g, r = cv2.split(img)
    # 对每个颜色通道分别使用灰度直方图均衡化方法去雾
    b_channel_equalized = cv2.equalizeHist(b)
    g_channel_equalized = cv2.equalizeHist(g)
    r_channel_equalized = cv2.equalizeHist(r)
    # 合并
    equalized_image = cv2.merge(
        (b_channel_equalized, g_channel_equalized, r_channel_equalized))
    cv2.imwrite(img_output, equalized_image)


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


if __name__ == "__main__":
    dehaze_test(
        input_path="./hazed-image",
        output_path="./dehazed-image/AOD-net"
    )
    dehaze_evaluate(
        input_path="./hazed-image",
        output_path="./dehazed-image/AOD-net"
    )

# 读取输入图像
# input_image = cv2.imread("./hazed-image/haze-3.jpg")

# 颜色通道拆分
# b, g, r = cv2.split(input_image)

# 对每个通道分别进行直方图均衡化
# b_channel_equal = cv2.equalizeHist(b)
# g_channel_equal = cv2.equalizeHist(g)
# r_channel_equal = cv2.equalizeHist(r)

# 合并均衡化后的三个通道
# equalized_image = cv2.merge(
    # (b_channel_equal, g_channel_equal, r_channel_equal))

# 保存去雾结果
# cv2.imwrite("./dehazed-image/equalized-image-3.jpg", equalized_image)

# 显示测试
# cv2.imshow("Original Image", input_image)
# cv2.imshow("Dehazed Image - Equalized", equalized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
