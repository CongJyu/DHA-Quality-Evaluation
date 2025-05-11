# 图像去雾质量评价 FR-IQA 实现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇


import numpy as np
import cv2
import os
import skimage
import pandas


def is_size_match(file_list_1, file_list_2):
    return len(file_list_1) == len(file_list_2)


# 评估均方误差 MSE
def get_mse(original_image_dir, dehazed_image_dir, result_save_dir):
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

    mse_result = []
    cols = ["Image File", "MSE"]

    for file_name in dehazed_image_list:
        original_image = cv2.imread(
            os.path.join(original_image_dir, file_name)
        )
        dehazed_image = cv2.imread(
            os.path.join(dehazed_image_dir, file_name)
        )

        # 转换成灰度图像
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2GRAY)
        # 计算 MSE
        current_mse = np.round(
            np.sum(
                cv2.subtract(original_image, dehazed_image) ** 2
            ) / float(original_image.shape[0] * original_image.shape[1]), 6
        )

        current_result = [file_name, str(current_mse)]
        mse_result.append(current_result)

        result = pandas.DataFrame(columns=cols, data=mse_result)
        print("[ DEBUG ] MSE Result CSV:\n", result)
        result.to_csv(
            os.path.join(result_save_dir, "FR_IQA_MSE.csv"),
            encoding="UTF-8"
        )


# 评估峰值信噪比 PSNR
def get_psnr(original_image_dir, dehazed_image_dir, result_save_dir):
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

        result = pandas.DataFrame(columns=cols, data=psnr_result)
        print("[ DEBUG ] PSNR Result CSV:\n", result)
        result.to_csv(
            os.path.join(result_save_dir, "FR_IQA_PSNR.csv"),
            encoding="UTF-8"
        )


# 评估结构相似度 SSIM
def get_ssim(original_image_dir, dehazed_image_dir, result_save_dir):
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

    ssim_result = []
    cols = ["Image File", "SSIM"]

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
        current_ssim = np.round(
            skimage.metrics.structural_similarity(
                original_image, dehazed_image,
                win_size=7, data_range=255, channel_axis=2
            ), 6
        )
        current_result = [file_name, str(current_ssim)]
        ssim_result.append(current_result)

        result = pandas.DataFrame(columns=cols, data=ssim_result)
        print("[ DEBUG ] SSIM Result CSV:\n", result)
        result.to_csv(
            os.path.join(result_save_dir, "FR_IQA_SSIM.csv"),
            encoding="UTF-8"
        )


if __name__ == "__main__":
    # 暗通道先验去雾
    get_mse(
        original_image_dir="./test-data-dcp/GT",
        dehazed_image_dir="./test-data-dcp/dehazed",
        result_save_dir="./test-data-dcp/evaluate"
    )
    get_psnr(
        original_image_dir="./test-data-dcp/GT",
        dehazed_image_dir="./test-data-dcp/dehazed",
        result_save_dir="./test-data-dcp/evaluate"
    )
    get_ssim(
        original_image_dir="./test-data-dcp/GT",
        dehazed_image_dir="./test-data-dcp/dehazed",
        result_save_dir="./test-data-dcp/evaluate"
    )
    # 快速恢复单色或灰度图像可见性
    get_mse(
        original_image_dir="./test-data-fvr/GT",
        dehazed_image_dir="./test-data-fvr/dehazed",
        result_save_dir="./test-data-fvr/evaluate"
    )
    get_psnr(
        original_image_dir="./test-data-fvr/GT",
        dehazed_image_dir="./test-data-fvr/dehazed",
        result_save_dir="./test-data-fvr/evaluate"
    )
    get_ssim(
        original_image_dir="./test-data-fvr/GT",
        dehazed_image_dir="./test-data-fvr/dehazed",
        result_save_dir="./test-data-fvr/evaluate"
    )
    # AOD-Net
    get_mse(
        original_image_dir="./test-data-aod/GT",
        dehazed_image_dir="./test-data-aod/dehazed",
        result_save_dir="./test-data-aod/evaluate"
    )
    get_psnr(
        original_image_dir="./test-data-aod/GT",
        dehazed_image_dir="./test-data-aod/dehazed",
        result_save_dir="./test-data-aod/evaluate"
    )
    get_ssim(
        original_image_dir="./test-data-aod/GT",
        dehazed_image_dir="./test-data-aod/dehazed",
        result_save_dir="./test-data-aod/evaluate"
    )
    # 直方图均衡化图像增强
    get_mse(
        original_image_dir="./test-data-hist/GT",
        dehazed_image_dir="./test-data-hist/dehazed",
        result_save_dir="./test-data-hist/evaluate"
    )
    get_psnr(
        original_image_dir="./test-data-hist/GT",
        dehazed_image_dir="./test-data-hist/dehazed",
        result_save_dir="./test-data-hist/evaluate"
    )
    get_ssim(
        original_image_dir="./test-data-hist/GT",
        dehazed_image_dir="./test-data-hist/dehazed",
        result_save_dir="./test-data-hist/evaluate"
    )
