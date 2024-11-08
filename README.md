### Automatic License Plate Recognition (ALPR)

This is a simple, efficient and aiming to be state-of-the art license plate recognition, i.e., detect and read license plates.


## Goals

1. Minimum requirements
2. Easy to change to the newest Pytorch
3. Should run *fast* on CPU
4. Easy to train

## Architecture

It is composed of two networks:

1. Detection: heavily inspired on [SCRFD](https://arxiv.org/abs/2105.04714). Changed the backbone and neck, however the main ideas and losses remains. It is finding both the plate bounding box using the Integral regression from [TinaFace](https://arxiv.org/abs/2011.13183) that is just a tidy slower but more accurate, and the plate corners with a simple regression head.
2. OCR: given a licence plate detection, I'm using [MaskOCR](https://arxiv.org/abs/2206.00311) to read it. It is one of the state of the art OCRs with a really good CPU runtime also.


Given an image, first we use the Detection network. Then, for every detected license plate, we crop it and send it to the OCR. Quite simple and somewhat fast.


## Datasets

I'm using the [Chinese City Parking Dataset](https://github.com/detectRecog/CCPD). It is quite big (over 200k images), and free to download by anyone. Notice that there are multiple scenarios within this dataset, but I'm just using the following:

- Base: train
- Weather: validation
- Challenge: test


## Install

Not quite an installation process, but to run this code you must run the following:
```sh
cd alpr/
python -m pip install -r requirements.txt
```

I recommend that you use Conda or a Virtualenv. To install Conda (as in [here](https://docs.anaconda.com/miniconda/)):
```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

And finally, using this lib with Conda:

```sh
cd alpr/
conda create --name alpr python=3.8
conda activate alpr
python -m pip install -r requirements.txt
```

## Train

TODO


## Test

TODO

## Benchmarks

TODO


## Alternative ideas that did not pan out

I tried to make a full end-to-end network, with both a detection, keypoint extractor and OCR within the same network. The OCR was a simple CRNN network, and I was reusing the same backbone and neck to extract the license plate features. Ideally this should reuse the same network weights (smaller end model), and act as a regularization, since the same backbone must do several different tasks at the same time.

It did not work that well. Both because CRNN is far from the current state of the art and because it is a tricky network to train. In the past I used to do curricular learning (i.e. start with the easiest examples and then train with the harder ones), but it was a little trickier this time. And since I was not in the mood to change the detection framework to use ViT, I decided to let go of this idea.