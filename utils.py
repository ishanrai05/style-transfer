from PIL import Image
import numpy as np

from torchvision import transforms 
import torch

# helper function to load the image and transform it into Tensors
# The paper uses an image of size 512x512

def image_loader(img_path, max_size=512, shape=None):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/10
    im_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.225, 0.225))
    ])
    
    # In images with an alpha channel, each pixel not only has a color value, but also has a numerical 
    # transparency value that defines what will happen when the pixel is placed over another pixel.
    
    # discard transparent alpha channela and add batch dimensions
    image = im_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


def tensor_to_image(tensor):
    
    # constructs a new view on a tensor which is declared not to need gradients,
    # i.e., it is to be excluded from further tracking of operations
    img = tensor.to("cpu").clone().detach()
    
    # convert the image to numpy and remove an extra dimension
    img = img.numpy().squeeze()
    
    img = img.transpose(1,2,0)
    
    # Unnormalize the image
    img = img * np.array((0.229, 0.224, 0.225)) +  np.array((0.485, 0.456, 0.406))
    
    img  = img.clip(0,1)
    
    return img

