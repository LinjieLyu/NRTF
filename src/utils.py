#!/usr/bin/env python3
import os
import pdb
import logging
import torch

from glob import glob
import pyredner

import numpy as np
import imageio
import skimage
from skimage.transform import rescale, resize, downscale_local_mean



def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def glob_imgs_hdr(path):
    imgs = []
    for ext in ['*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    
    return imgs


def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def load_rgb(path,target_res):
    envmap_img=pyredner.imread(path,gamma=1)[:,:,:3]
    
    img = envmap_img.permute(2, 0, 1) # HWC -> CHW
    img = img.unsqueeze(0) # CHW -> NCHW
    img = torch.nn.functional.interpolate(img, size = target_res, mode = 'area')
    img = img.squeeze(dim = 0) # NCHW -> CHW
    img = img.permute(1, 2, 0)
    envmap_img_lowres = img
  
    return envmap_img_lowres

def load_mask(path,target_res):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    alpha_resized = resize(alpha, target_res,
                       anti_aliasing=True)
    object_mask = alpha_resized > 0.5
   
    return object_mask