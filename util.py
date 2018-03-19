import os
import scipy.misc as scm
import numpy as np


def get_image(img, flip=False): # [0,255] to [-1,1]
    if flip:
        img = np.fliplr(img)
    img = img * 2. /255. - 1.
    img = img[..., ::-1]  # rgb to bgr
    return img

def get_label(label):
    one_hot = np.zeros(size)
    one_hot[ label ] = 1.0
    one_hot[ one_hot==0 ] = 0.0
    return one_hot

def inverse_image(img): # [-1,1] to [0,255]
    img = (img + 1.) / 2. * 255.
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[..., ::-1] # bgr to rgb
    return img

def get_shape_c(tensor): # static shape
    return tensor.get_shape().as_list()