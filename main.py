# 图像去雾的质量评价
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

# Use Python 3.10, with package manager and project manager `uv`.

import dark_channel_prior
import fast_visibility_restoration
import equalized_hist


# 测试数据存放目录
input_path = "./hazed-image"
output_path_1 = "./dehazed-image-dark-channel-prior"
output_path_2 = "./dehazed-image-fast-vis-restoration"
output_path_3 = "./dehazed-image-equalized-hist"


def main():
    # Kaiming He 暗通道去雾方法测试
    print("\n[ TEST ] Dark Channel Prior Dehazing\n")
    dark_channel_prior.dehaze_test(input_path, output_path_1)
    print("\n[ TEST ] Dark Channel Prior Evaluating\n")
    dark_channel_prior.dehaze_evaluate(input_path, output_path_1)
    # Tarel 快速恢复可见性方法测试
    print("\n[ TEST ] Fast Visibility Restoration Dehazing\n")
    fast_visibility_restoration.dehaze_test(input_path, output_path_2)
    print("\n[ TEST ] Fast Visibility Restoration Evaluating\n")
    fast_visibility_restoration.dehaze_evaluate(input_path, output_path_2)
    # 图像增强方式的灰度直方图均衡方法测试
    print("\n[ TEST ] Equalize Histogram Dehazing\n")
    equalized_hist.dehaze_test(input_path, output_path_3)
    print("\n[ TEST ] Equalize Histogram Evaluating\n")
    equalized_hist.dehaze_evaluate(input_path, output_path_3)


if __name__ == "__main__":
    main()
