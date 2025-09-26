# The Quality Evaluation of Image Dehazing Algorithms
# Rain CongJyu CHEN

# Use Python 3.10, with package manager and project manager `uv`.

import os

if __name__ == "__main__":
    print("[ INFO ] This is the DHA-Quality-Evaluation project.")

    print("[ INFO ] Using Dark Channel Prior Method for Dehazing.")
    os.system("uv run dark_channel_prior.py")
    print("[ INFO ] Using Fast Visibility Restoration Method for Dehazing.")
    os.system("uv run fast_vis_restoration.py")
    print("[ INFO ] Using AOD-Net for Dehazing.")
    os.system("uv run AOD_dehaze.py")
    print("[ INFO ] Using Histogram Equalization Method for Dehazing.")
    os.system("uv run equalized_hist.py")
    print("[ INFO ] Use FR-IQA for Evaluation Process.")
    os.system("uv run FR_IQA.py")
    print("[ INFO ] All Process Done.")
