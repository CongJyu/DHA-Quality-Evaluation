import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
# import torch.optim
# import os
# import sys
# import argparse
# import time
import AOD_dataloader
import AOD_net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


def dehaze_image(image_path):

    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy)/255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cpu().unsqueeze(0)

    dehaze_net = AOD_net.dehaze_net().cpu()
    dehaze_net.load_state_dict(torch.load('AOD-net-snapshots/Epoch9.pth'))

    clean_image = dehaze_net(data_hazy)
    # torchvision.utils.save_image(torch.cat(
    #     (data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])
    torchvision.utils.save_image(
        clean_image, "dehazed-image-AOD-net/" + image_path.split("/")[-1]
    )


if __name__ == '__main__':

    test_list = glob.glob("hazed-image/*")

    for image in test_list:

        print("[ INFO ] Processing image: ", image)
        dehaze_image(image)
        # print(image, "done!")

    print("[ INFO ] Success. All images done.")
