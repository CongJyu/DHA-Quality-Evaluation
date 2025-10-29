# Reproduction of Restore Visibility from Single Colour or Gray-Scale Images Method
# Rain CongJyu CHEN

# Import computer vision libraries.
import cv2
import numpy as np
import os

# TODO: This method needs further improvement.


# Make white balance process.
def white_balance(image):
    """
    :param image: image data read.
    :balanced_image: white balanced image return.
    """
    red, green, blue = cv2.split(image)
    red_avg = cv2.mean(red)[0]
    green_avg = cv2.mean(green)[0]
    blue_avg = cv2.mean(blue)[0]
    # Gain of each channels.
    k = (red_avg + green_avg + blue_avg) / 3
    k_red = k / red_avg
    k_green = k / green_avg
    k_blue = k / blue_avg
    # red = red * k_red
    # green = green * k_green
    # blue = blue * k_blue
    red = cv2.addWeighted(src1=red, alpha=k_red, src2=0, beta=0, gamma=0)
    green = cv2.addWeighted(src1=green, alpha=k_green, src2=0, beta=0, gamma=0)
    blue = cv2.addWeighted(src1=blue, alpha=k_blue, src2=0, beta=0, gamma=0)
    balanced_image = cv2.merge([red, green, blue])
    return balanced_image


def white_balance_old(img_input):
    """
    :param img: read image data from `cv2.imread`.
    :return: white balanced image return.
    """
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / avg_b
            g = int(img[i][j][1]) * maxvalue / avg_g
            r = int(img[i][j][2]) * maxvalue / avg_r
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r

    return img


# Get dark channel.
def get_dark_channel(img, size=20):
    # `size` is the size of window. `size` is larger then the probability of the dark channel is larger,
    # and the channel is darker.
    r, g, b = cv2.split(img)  # Split image into three channels.
    min_channel = cv2.min(r, cv2.min(g, b))  # Fetch the minimum channel.
    # Rectangle Kernel.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_img = cv2.erode(min_channel, kernel)
    dark_channel_img = cv2.merge([dark_channel_img, dark_channel_img, dark_channel_img])
    return dark_channel_img


# Predict atmosphere veil (AV).
# Calculate W(x, y) = min(I(x, y)), and get the minimum channel.
def get_min_channel(image):
    red, green, blue = cv2.split(image)
    min_channel = cv2.min(red, cv2.min(green, blue))
    return min_channel


def get_min_channel_fix(image):
    img_gray = np.zeros((image.shape[0], image.shape[1]), np.float32)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            localMin = 255
            for k in range(0, 2):
                if image.item((i, j, k)) < localMin:
                    localMin = image.item((i, j, k))
            img_gray[i, j] = localMin
    return img_gray


# Get atmosphere veil (AV).
def get_atmos_veil(image, sv=41, p=0.95):
    # `sv` is the rectangle window's size of the median filter.
    # Factor `p` ranges from [0, 1], controlling the strength of the restoration of the visibility.
    w_xy = get_min_channel_fix(image)
    a_xy = cv2.medianBlur(np.uint8(w_xy), sv)
    b_xy = w_xy - a_xy
    b_xy = np.abs(b_xy)
    b_xy = a_xy - cv2.medianBlur(np.uint8(b_xy), sv)
    max_255_img = np.ones(b_xy.shape, dtype=np.uint8) * 255
    min_t = cv2.merge([np.uint8(p * b_xy), np.uint8(w_xy), max_255_img])
    min_t = get_min_channel_fix(min_t)
    min_t[min_t < 0] = 0
    veil = np.uint8(min_t)
    veil = cv2.blur(veil, (5, 5))
    # w_xy = get_min_channel_fix(image)
    # a_xy = cv2.medianBlur(np.uint8(w_xy), sv)
    # b_xy = np.abs(w_xy - a_xy)
    # b_xy = a_xy - cv2.medianBlur(np.uint8(b_xy), sv)
    # max_image = np.ones(b_xy.shape, dtype=np.uint8) * 255
    # # min_image = cv2.merge(
    # #     [np.uint8(p * b_xy), np.uint8(w_xy), max_image]
    # # )
    # min_image = cv2.min(np.uint8(p * b_xy), cv2.min(np.uint8(w_xy), max_image))
    # # print("[ DEBUG ] min_image's shape:", min_image.shape)
    # # min_image = get_min_channel_fix(min_image)
    # # min_image = get_min_channel(min_image)
    # min_image[min_image < 0] = 0
    # # cv2.imshow("min_image", min_image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # veil = np.uint8(min_image)
    # veil = cv2.blur(veil, (5, 5))
    # veil = np.float32(veil) / 255
    return veil


def color_correct(img, u):
    img = np.float64(img) / 255
    B_mse = np.std(img[:, :, 0])
    G_mse = np.std(img[:, :, 1])
    R_mse = np.std(img[:, :, 2])

    B_max = np.mean(img[:, :, 0]) + u * B_mse
    G_max = np.mean(img[:, :, 1]) + u * G_mse
    R_max = np.mean(img[:, :, 2]) + u * R_mse

    B_min = np.mean(img[:, :, 0]) - u * B_mse
    G_min = np.mean(img[:, :, 1]) - u * G_mse
    R_min = np.mean(img[:, :, 2]) - u * R_mse

    B_cr = (img[:, :, 0] - B_min) / (B_max - B_min)
    G_cr = (img[:, :, 1] - G_min) / (G_max - G_min)
    R_cr = (img[:, :, 2] - R_min) / (R_max - R_min)

    img_CR = cv2.merge([B_cr, G_cr, R_cr]) * 255
    img_CR = np.clip(img_CR, 0, 255)
    img_CR = np.uint8(img_CR)

    return img_CR


# Dehaze.
def dehaze(img_input, img_output):
    original_image = cv2.imread(img_input)
    white_balanced_img = white_balance(original_image)
    veil = get_atmos_veil(white_balanced_img)
    veil = np.float32(veil) / 255
    dehazed_img = np.zeros((veil.shape[0], veil.shape[1], 3), dtype=np.float32)
    white_balanced_img = np.float32(white_balanced_img) / 255
    for i in range(3):
        dehazed_img[:, :, i] = (white_balanced_img[:, :, i] - veil) / (1 - veil)
    dehazed_img = dehazed_img / dehazed_img.max()
    dehazed_img = np.clip(dehazed_img, 0, 1)
    dehazed_img = np.uint8(dehazed_img * 255)
    dehazed_img = color_correct(dehazed_img, 2)
    cv2.imwrite(img_output, dehazed_img)


# Test the dataset.
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


# Evaluate the dataset.
# def dehaze_evaluate(input_path, output_path):
#     input_path_list = os.listdir(input_path)
#     if ".DS_Store" in input_path_list:
#         input_path_list.remove(".DS_Store")
#     elif input_path_list is not None:
#         print("[ FAIL ] Original image path is empty.")
#     output_path_list = os.listdir(output_path)
#     if ".DS_Store" in output_path_list:
#         output_path_list.remove(".DS_Store")
#     elif output_path_list is not None:
#         print("[ FAIL ] Dehazed image path is empty.")

#     print("Image    \tPSNR     \tSSIM\n---------\t---------\t---------")
#     for file_name in input_path_list:
#         # original_image_path = os.path.join(input_path, file_name)
#         original_image_path = os.path.join("./clear-image", file_name)
#         dehazed_image_path = os.path.join(output_path, file_name)
#         original_image = cv2.imread(original_image_path)
#         original_image = original_image.astype("float32") / 255
#         dehazed_image = cv2.imread(dehazed_image_path)
#         dehazed_image = dehazed_image.astype("float32") / 255
#         current_psnr = round(evaluation.compare_psnr(
#             original_image, dehazed_image), 6)
#         current_ssim = round(evaluation.compare_ssim(
#             original_image, dehazed_image, win_size=7, data_range=255, channel_axis=2), 6)
#         print(file_name, "\t", current_psnr, "\t", current_ssim)


# Store the image of atmosphere veil.
def save_veil(input_path, output_path):
    input_path_list = os.listdir(input_path)
    if ".DS_Store" in input_path_list:
        input_path_list.remove(".DS_Store")
    output_path_list = os.listdir(output_path)
    if ".DS_Store" in output_path_list:
        output_path_list.remove(".DS_Store")

    for file_name in input_path_list:
        input_hazed_image = os.path.join(input_path, file_name)
        output_dehazed_image = os.path.join(output_path, file_name)
        print("[ INFO ] Getting veil for image: ", file_name)
        img = cv2.imread(input_hazed_image)
        veil_image = get_atmos_veil(img)
        cv2.imwrite(output_dehazed_image, veil_image)


# Store the image of white balanced immage.
def save_white_balanced_image(input_path, output_path):
    input_path_list = os.listdir(input_path)
    if ".DS_Store" in input_path_list:
        input_path_list.remove(".DS_Store")
    output_path_list = os.listdir(output_path)
    if ".DS_Store" in output_path_list:
        output_path_list.remove(".DS_Store")

    for file_name in input_path_list:
        input_hazed_image = os.path.join(input_path, file_name)
        output_dehazed_image = os.path.join(output_path, file_name)
        print("[ INFO ] Generating white balanced image: ", file_name)
        img = cv2.imread(input_hazed_image)
        white_balance_image = white_balance(img)
        cv2.imwrite(output_dehazed_image, white_balance_image)


if __name__ == "__main__":
    save_white_balanced_image(
        input_path="./test-data-fvr/hazy", output_path="./test-data-fvr/white-balanced"
    )
    save_veil(input_path="./test-data-fvr/hazy", output_path="./test-data-fvr/veil")
    dehaze_test(
        input_path="./test-data-fvr/hazy", output_path="./test-data-fvr/dehazed"
    )
    # dehaze_test(
    #     input_path="./hazed-image",
    #     output_path="./dehazed-image/fast-vis-restoration"
    # )
