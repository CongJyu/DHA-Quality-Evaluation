# 图像去雾质量评价 PSNR 峰值信噪比指标 与 SSIM 结构相似性指标 实现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

# PSNR 是基于均方误差的一种度量方式，用来评估图像还原的质量，高的 PSNR 值表明图像与原图相比差异较小，适用于评价去雾后图像的整体质量和保真度

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# 比较去雾后图像和原图的 PSNR
# 函数返回去雾后图像和原图的峰值信噪比
# def compare_psnr(original_image, dehazed_image, max_value=255):
# original_image = original_image.astype(np.float64)
# dehazed_image = dehazed_image.astype(np.float64)
# mse = np.mean((original_image - dehazed_image) ** 2)
# psnr = 10 * np.log10((max_value ** 2) / mse)
# return psnr


# 比较去雾后图像和原图的 SSIM
# 函数返回去雾后图像和原图的结构相似性
# def compare_ssim(original_image, dehazed_image):
