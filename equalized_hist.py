# Reproduction of Histogram Equalization Dehazing Method
# Rain CongJyu CHEN

import os

import cv2


# Traditional image enhancement methods.
def dehaze(img_input, img_output):
    # Read foggy image (color).
    img = cv2.imread(img_input)
    # Split color channels.
    b, g, r = cv2.split(img)
    # Use grayscale histogram equalization method to remove fog for each color channel.
    b_channel_equalized = cv2.equalizeHist(b)
    g_channel_equalized = cv2.equalizeHist(g)
    r_channel_equalized = cv2.equalizeHist(r)
    # Merge.
    equalized_image = cv2.merge(
        (b_channel_equalized, g_channel_equalized, r_channel_equalized)
    )
    cv2.imwrite(img_output, equalized_image)


# Test datasets.
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


# Evaluate datasets.
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


if __name__ == "__main__":
    dehaze_test(
        input_path="./test-data-hist/hazy", output_path="./test-data-hist/dehazed"
    )
    # dehaze_evaluate(
    #     input_path="./hazed-image",
    #     output_path="./dehazed-image/equalized-hist"
    # )

# Read input image.
# input_image = cv2.imread("./hazed-image/haze-3.jpg")

# Color channel splitting.
# b, g, r = cv2.split(input_image)

# Perform histogram equalization on each channel separately.
# b_channel_equal = cv2.equalizeHist(b)
# g_channel_equal = cv2.equalizeHist(g)
# r_channel_equal = cv2.equalizeHist(r)

# Merge the three equalized channels.
# equalized_image = cv2.merge(
# (b_channel_equal, g_channel_equal, r_channel_equal))

# Save the dehazing results.
# cv2.imwrite("./dehazed-image/equalized-image-3.jpg", equalized_image)

# Display tests.
# cv2.imshow("Original Image", input_image)
# cv2.imshow("Dehazed Image - Equalized", equalized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
