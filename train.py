import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_features, gram_matrix, tensor_to_image

def train(optimizer, model, target, content_features, style_weights, style_grams, content_weight, style_weight, steps=2000, show_every=200):

    for ii in tqdm(range(1, steps+1)):
        
        # get the features from your target image
        target_features = get_features(target, model)

        # the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
            
        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(tensor_to_image(target))
            plt.show()
    return target