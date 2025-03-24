import dark_channel_prior
import fast_visibility_restoration

input_path = "./hazed-image"
output_path_1 = "./dehazed-image-dark-channel-prior"
output_path_2 = "./dehazed-image-fast-vis-restoration"


def main():
    print("\n[ TEST ] Dark Channel Prior Dehazing\n")
    dark_channel_prior.test(input_path, output_path_1)
    print("\n[ TEST ] Fast Visibility Restoration Dehazing\n")
    fast_visibility_restoration.test(input_path, output_path_2)


if __name__ == "__main__":
    main()
