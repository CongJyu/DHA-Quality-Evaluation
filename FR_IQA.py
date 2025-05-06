# 图像去雾质量评价 PSNR 峰值信噪比指标 与 SSIM 结构相似性指标 实现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

# PSNR 是基于均方误差的一种度量方式，用来评估图像还原的质量，高的 PSNR 值表明图像与原图相比差异较小，适用于评价去雾后图像的整体质量和保真度

import numpy as np
import cv2
import os
import skimage
import pandas
import csv


def is_size_match(file_list_1, file_list_2):
    return len(file_list_1) == len(file_list_2)


# 评估均方误差 MSE
def get_mse(original_image_dir, dehazed_image_dir):
    original_image_list = os.listdir(original_image_dir)
    if ".DS_Store" in original_image_list:
        original_image_list.remove(".DS_Store")
    elif original_image_list is None:
        print("[ FAIL ] Original image dir is empty.")
        return -1
    dehazed_image_list = os.listdir(dehazed_image_dir)
    if ".DS_Store" in dehazed_image_list:
        dehazed_image_list.remove(".DS_Store")
    elif dehazed_image_list is None:
        print("[ FAIL ] Original image path is empty.")
        return -1

    # 检查去雾后的图像个数与原无雾图像个数是否对应
    if is_size_match(original_image_list, dehazed_image_list):
        print("[ INFO ] File list check passed.")
    else:
        print("[ FAIL ] File list is not matched. Stop.")
        return -1

    for file_name in dehazed_image_list:
        original_image = cv2.imread(
            os.path.join(original_image_dir, file_name)
        )
        dehazed_image = cv2.imread(
            os.path.join(dehazed_image_dir, file_name)
        )


# 评估峰值信噪比 PSNR
def get_psnr(original_image_dir, dehazed_image_dir):
    original_image_list = os.listdir(original_image_dir)
    if ".DS_Store" in original_image_list:
        original_image_list.remove(".DS_Store")
    elif original_image_list is None:
        print("[ FAIL ] Original image dir is empty.")
        return -1
    dehazed_image_list = os.listdir(dehazed_image_dir)
    if ".DS_Store" in dehazed_image_list:
        dehazed_image_list.remove(".DS_Store")
    elif dehazed_image_list is None:
        print("[ FAIL ] Original image path is empty.")
        return -1

    psnr_result = []
    cols = ["Image File", "PSNR"]

    # 检查去雾后的图像个数与原无雾图像个数是否对应
    if is_size_match(original_image_list, dehazed_image_list):
        print("[ INFO ] File list check passed.")
    else:
        print("[ FAIL ] File list is not matched. Stop.")
        return -1

    for file_name in dehazed_image_list:
        original_image = cv2.imread(
            os.path.join(original_image_dir, file_name)
        )
        dehazed_image = cv2.imread(
            os.path.join(dehazed_image_dir, file_name)
        )
        current_psnr = np.round(
            skimage.metrics.peak_signal_noise_ratio(
                original_image, dehazed_image,
            ), 6
        )
        current_result = [file_name, str(current_psnr)]
        psnr_result.append(current_result)

    with open("FR-IQA-PSNR.csv", mode="w", newline="") as result_file:
        result = pandas.DataFrame(columns=cols, data=psnr_result)
        print("[ DEBUG ] Result CSV:\n", result)
        result.to_csv("FVR_FR_IQA_PSNR.csv", encoding="UTF-8")


if __name__ == "__main__":
    get_psnr(
        original_image_dir="./test-data-fvr/GT",
        dehazed_image_dir="./test-data-fvr/dehazed"
    )
