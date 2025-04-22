# AOD-Net 端到端去雾网络 复现
# 湖南大学 信息科学与工程学院 通信工程 陈昶宇

# 参考 https://github.com/MayankSingal/PyTorch-Image-Dehazing 代码进行修改

# AOD-Net 去雾程序

import torch
import torchvision
import numpy as np
from PIL import Image
import glob


# 设置训练 AOD-Net 使用的设备
if torch.mps.is_available():
    training_device = torch.device("mps")
elif torch.cuda.is_available():
    training_device = torch.device("cuda")
else:
    training_device = torch.device("cpu")


# AOD_net
class aod_dehaze_net(torch.nn.Module):
    def __init__(self):
        super(aod_dehaze_net, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.e_conv1 = torch.nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = torch.nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = torch.nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = torch.nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = torch.nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))
        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image


def dehaze_image(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    # data_hazy = data_hazy.cpu().unsqueeze(0)
    data_hazy = data_hazy.to(training_device).unsqueeze(0)
    # dehaze_net = AOD_net.dehaze_net().cpu()
    dehaze_net = aod_dehaze_net().to(training_device)
    dehaze_net.load_state_dict(torch.load('AOD-net-snapshots/dehazer.pth'))
    clean_image = dehaze_net(data_hazy)
    # torchvision.utils.save_image(torch.cat(
    #     (data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])
    torchvision.utils.save_image(
        clean_image, "dehazed-image-AOD-net/" + image_path.split("/")[-1]
    )


if __name__ == '__main__':
    # 设备提示
    if torch.mps.is_available():
        print("[ INFO ] Start process with MPS.\n")
    elif torch.cuda.is_available():
        print("[ INFO ] Start process with CUDA\n")
    else:
        print("[ INFO ] Start process with CPU\n")
    test_list = glob.glob("hazed-image/*")
    for image in test_list:
        print("[ INFO ] Processing image: ", image)
        dehaze_image(image)
        # print(image, "done!")
    print("[ INFO ] Success. All images done.")
