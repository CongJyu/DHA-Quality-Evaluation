# The Reproduction of the AOD-Net End-to-End Dehazing Network
# Rain CongJyu CHEN

# Refs: https://github.com/MayankSingal/PyTorch-Image-Dehazing

# AOD-Net Training Program

import torch
import torchvision
import os
import argparse
import time
import numpy as np
from PIL import Image
import glob
import random

# AOD_dataloader
random.seed(2025)
# Set device for training AOD-Net.
if torch.mps.is_available():
    training_device = torch.device("mps")
elif torch.cuda.is_available():
    training_device = torch.device("cuda")
else:
    training_device = torch.device("cpu")


def populate_train_list(orig_images_path, hazy_images_path):
    # Initialize two lists.
    train_list = []
    val_list = []
    # Get all hazed images.
    # Use `glob` to get jpg files from certain directory.
    image_list_haze = glob.glob(hazy_images_path + "*.jpg")
    # Build mapping.
    tmp_dict = {}
    # Extract key word.
    for image in image_list_haze:
        image = image.split("/")[-1]
        # image = image.split("/")[-1][5:]  # changed
        key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
        # Store mapping of hazed images.
        # `key` is the name of clear images.
        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)
    # Divide training set and testing set.
    # Training set is put in list `train_keys`.
    # Testing set is put in list `val_keys`.
    train_keys = []
    val_keys = []
    len_keys = len(tmp_dict.keys())
    # 90% of the data for training, and 10% for testing and verifying.
    for i in range(len_keys):
        if i < len_keys * 9 / 10:
            train_keys.append(list(tmp_dict.keys())[i])
        else:
            val_keys.append(list(tmp_dict.keys())[i])
    for key in list(tmp_dict.keys()):
        if key in train_keys:
            for hazy_image in tmp_dict[key]:
                # `train_list` is the path to clear images and hazed images.
                train_list.append(
                    [orig_images_path + key, hazy_images_path + hazy_image]
                )
        else:
            for hazy_image in tmp_dict[key]:
                # `val_list` is the path to clear images and hazed images.
                val_list.append([orig_images_path + key, hazy_images_path + hazy_image])
    # Shuffle the training set and the testing set.
    random.shuffle(train_list)
    random.shuffle(val_list)
    # Return the final dataset list.
    return train_list, val_list


class dehazing_loader(torch.utils.data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode="train"):
        self.train_list, self.val_list = populate_train_list(
            orig_images_path, hazy_images_path
        )
        if mode == "train":
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]
        data_orig = Image.open(data_orig_path)
        data_hazy = Image.open(data_hazy_path)
        # data_orig = data_orig.resize((480, 640), Image.ANTIALIAS)
        # data_hazy = data_hazy.resize((480, 640), Image.ANTIALIAS)
        data_orig = data_orig.resize((480, 640), Image.Resampling.LANCZOS)
        data_hazy = data_hazy.resize((480, 640), Image.Resampling.LANCZOS)
        data_orig = np.asarray(data_orig) / 255.0
        data_hazy = np.asarray(data_hazy) / 255.0
        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()
        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    # dehaze_net = AOD_net.dehaze_net().cpu()
    dehaze_net = aod_dehaze_net().to(training_device)
    dehaze_net.apply(weights_init)
    train_dataset = dehazing_loader(config.orig_images_path, config.hazy_images_path)
    val_dataset = dehazing_loader(
        config.orig_images_path, config.hazy_images_path, mode="val"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    # criterion = torch.nn.MSELoss().cpu()
    criterion = torch.nn.MSELoss().to(training_device)
    optimizer = torch.optim.Adam(
        dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    dehaze_net.train()
    for epoch in range(config.num_epochs):
        print(f"\n[ INFO ] EPOCH #{epoch}")
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            # img_orig = img_orig.cpu()
            # img_haze = img_haze.cpu()
            img_orig = img_orig.to(training_device)
            img_haze = img_haze.to(training_device)
            clean_image = dehaze_net(img_haze)
            loss = criterion(clean_image, img_orig)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dehaze_net.parameters(), config.grad_clip_norm
            )
            optimizer.step()
            if ((iteration + 1) % config.display_iter) == 0:
                print(
                    f"[ INFO ] Current epoch: {epoch}",
                    ", Loss at iteration",
                    iteration + 1,
                    ":",
                    loss.item(),
                )
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(
                    dehaze_net.state_dict(),
                    config.snapshots_folder + "Epoch" + str(epoch) + ".pth",
                )
        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            # img_orig = img_orig.cpu()
            # img_haze = img_haze.cpu()
            img_orig = img_orig.to(training_device)
            img_haze = img_haze.to(training_device)
            clean_image = dehaze_net(img_haze)
            torchvision.utils.save_image(
                torch.cat((img_haze, clean_image, img_orig), 0),
                config.sample_output_folder + str(iter_val + 1) + ".jpg",
            )
        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")


if __name__ == "__main__":
    # Notice for devices.
    if torch.mps.is_available():
        print("[ INFO ] Start process with MPS.\n")
    elif torch.cuda.is_available():
        print("[ INFO ] Start process with CUDA\n")
    else:
        print("[ INFO ] Start process with CPU\n")
    # Training time start.
    start_time = time.time()
    parser = argparse.ArgumentParser()
    # Input arguments and parametres.
    parser.add_argument(
        "--orig_images_path", type=str, default="training-image-AOD-net/images/"
    )
    parser.add_argument(
        "--hazy_images_path", type=str, default="training-image-AOD-net/data/"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--display_iter", type=int, default=10)
    parser.add_argument("--snapshot_iter", type=int, default=200)
    parser.add_argument("--snapshots_folder", type=str, default="AOD-net-snapshots/")
    parser.add_argument("--sample_output_folder", type=str, default="samples/")
    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)
    train(config)
    # Training time end.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[ INFO ] Training done. Elapsed time is {elapsed_time:.2f} second(s).")
