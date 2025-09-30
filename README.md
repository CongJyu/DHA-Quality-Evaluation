# The Reproduction and Quality Evaluation of Image Dehazing Algorithms

## Introduction

This repository contains the Python code of reproducton of some image dehazing algorithms (DHA), including:

- DCP (Dark Channel Prior Dehazing Method)
- FVR (Fast Visibility Restoration Method)
- AOD-Net (All-in-One Dehazing Network Method)

For each dehazing method there is a Python file to implement the algorithm.

## Usage

### Dataset Directory

To run this program correctly, you need to create several folders to store the datasets. In the root of the project
directory, create the following directories:

```
.
├── AOD-net-snapshots
├── samples
├── test-data-aod
│   ├── dehazed
│   ├── evaluate
│   ├── GT
│   └── hazy
├── test-data-dcp
│   ├── dark-channel-prior
│   ├── dehazed
│   ├── evaluate
│   ├── GT
│   └── hazy
├── test-data-fvr
│   ├── dehazed
│   ├── evaluate
│   ├── GT
│   ├── hazy
│   ├── veil
│   └── white-balanced
├── test-data-hist
│   ├── dehazed
│   ├── evaluate
│   ├── GT
│   └── hazy
└── training-image-AOD-net
    ├── data
    └── images
```

Put the hazy or not dehazed image in the `hazy/` directory, while put the clear image (also known as "GT") in `GT/`
directory if you have prepared the images for evaluation. Specifically for the AOD-Net training, put the original clear
image in `training-image-AOD-net/images/` directory, and put the hazy image in `training-image-AOD-net/data/` directory.
After the training of AOD-Net, the model will be stored in `AOD-net-snapshots` directory.

### Environment Configuration

We use `uv` to manage this project, so we recommend you to install a copy of `uv` in your system to handle the Python
projects. Here are the steps.

#### Install the proper version of Python

```shell
uv python install 3.13
```

#### Create a virtual environment

If you use PyCharm, you can simply create a virtual environment for this project in GUI through `uv`; if you use other
IDEs, you can also use `uv` command line to create a virtual environment.

This project is based on Python 3.13, so just install the specific verison if you don't have Python 3.13 installed in
your computer.

```shell
uv venv --python 3.13
```

#### Install the requirements

Auto install the requirements in your IDE, or install manually:

```shell
uv pip install [package_name]
```

#### Start training AOD-Net

Run the `AOD_train.py` to train the AOD-Net model for image dehazing. Click the "run single file" button or:

```shell
uv run AOD_train.py
```

This process may take a little long time. The program will exit automatically when the training task is finished.

#### Test dehazing methods

Run single file or run all methods at a time, the evaluation result will be stored in `evaluate/` directory as `csv`
files.
