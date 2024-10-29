from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import os
import pandas as pd
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


######################################################################
# Now the style loss module looks almost exactly like the content loss
# module. The style distance is also computed using the mean square
# error between :math:`G_{XL}` and :math:`G_{SL}`.
#

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()



######################################################################
# Additionally, VGG networks are trained on images with each channel
# normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We will use them to normalize the image before sending it into the network.
#

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_style_content_loss(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(False)
    model.requires_grad_(False)


    # correct the values of updated input image
    with torch.no_grad():
        input_img.clamp_(0, 1)

    model(input_img)
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss


    loss = (style_score * style_weight)+ (content_score * content_weight)
    print(f'Style Loss: {style_score} \t Content Loss: {content_score} \t Total Loss: {loss}')
    return style_score.detach().cpu(), content_score.detach().cpu(), loss.detach().cpu()
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'StyleLoss',
                    description = 'Takes the path to two images and gives Style and Content Loss. 0 means identical')
    parser.add_argument('--original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--edited_path', help='path to edited image', type=str, required=True)
    parser.add_argument('--promtps_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument(
        "--image",
        action="store_true",
        help="Whether it is a single image path",
    )

    args = parser.parse_args()
    if args.image:
        #read original and edited images
        original = image_loader(args.original_path)
        edited = image_loader(args.edited_path)
        # compute style, content and total score
        style_score, content_score, total_score = get_style_content_loss(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img=original, style_img=original, input_img=edited)
        # save the results
        if args.save_path is not None:
            df = pd.DataFrame({'filename': [args.edited_path.split('/')[-1]], 'Style_Loss': [style_score], 'Content_Loss': [content_score], 'Total_Loss': [total_score]})
            df.to_csv(args.save_path)
    else:
        # read the image filenames in the folder
        file_names = os.listdir(args.original_path)
        file_names = [name for name in file_names if '.png' in name]
        # read the prompts csv
        df_prompts = pd.read_csv(args.promtps_path)
        # initialise columns to store the losses
        df_prompts['style_loss'] = df_prompts['case_number'] *0
        df_prompts['content_loss'] = df_prompts['case_number']*0
        df_prompts['total_loss'] = df_prompts['case_number']*0
        # iter through all the rows
        for index, row in df_prompts.iterrows():
            # get case number for image reference
            case_number = row.case_number
            files = [file for file in file_names if file.startswith(f'{case_number}_')]
            style_scores = []
            content_scores = []
            total_scores = []
            for file in files:
                # read original image and corresponding edited image 
                original = image_loader(os.path.join(args.original_path,file))
                edited = image_loader(os.path.join(args.edited_path,file))
                # compute losses
                style_score, content_score, total_score = get_style_content_loss(cnn, cnn_normalization_mean, cnn_normalization_std,
                                            content_img=original, style_img=original, input_img=edited)
            
                style_scores.append(style_score)
                content_scores.append(content_score)
                total_scores.append(total_score)
            # store the results in csv (by averaging it over images per prompt)
            df_prompts.loc[index,'style_loss'] = np.mean(style_scores)
            df_prompts.loc[index,'content_loss'] = np.mean(content_scores)
            df_prompts.loc[index,'total_loss'] = np.mean(total_scores)
        if args.save_path is not None:
            df_prompts.to_csv(os.path.join(args.save_path, f'{os.path.basename(args.edited_path)}_styleloss.csv'))
