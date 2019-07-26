import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, models

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils import image_loader, tensor_to_image, get_features, gram_matrix
from train import train

# Downloading the vgg19 modal and freezing all the parameters

device = torch.device("cpu")


parser = argparse.ArgumentParser(description='Style Transfer')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--show_img', type=bool, default=True, help='See sample images')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='train the model')
parser.add_argument('--content_path', default='images/content/main-building.jpg', type=str, help='path to content')
parser.add_argument('--style_path', default='images/style/starry-night.jpg', type=str, help='path to style')
parser.add_argument('--content_weight', default=1, type=int, help='content weight: alpha')
parser.add_argument('--style_weight', default=1e6, type=int, help='style weight: beta')
parser.add_argument('--show_every', default=200, type=int, help='displaying the target image, intermittently')
parser.add_argument('--steps', default=2000, type=int, help='iterations to update image')
opt = parser.parse_args()



if opt.use_cuda:
    # checking if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load in content and style image
content = image_loader(opt.content_path, shape=(512,512)).to(device)
style = image_loader(opt.style_path, shape=(512,512)).to(device)

# display images
if opt.show_img:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.imshow(tensor_to_image(content))
    ax2.imshow(tensor_to_image(style))



# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = opt.content_weight  # alpha
style_weight = opt.style_weight  # beta

if opt.train:
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device)

    for params in vgg.parameters():
        params.requires_grad_(False)

    # get content and style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # create a third "target" image and prep it for change
    # it is a good idea to start off with the target as a copy of our *content* image
    # then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)


    optimizer = optim.Adam([target], lr=0.003)

    # for displaying the target image, intermittently
    show_every = opt.show_every

    # iteration hyperparameters
    steps = opt.steps  # decide how many iterations to update your image

    output = train(optimizer, vgg, target, content_features, style_weights, style_grams, content_weight,  style_weight, steps, show_every)

    plt.imshow(tensor_to_image(output))

    plt.savefig('output.png')
