
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

import cv2



def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# ***** Features extraction *****

def extract_Sobel_filter(data_in): 
    """ Function to run Sobel filters (x and y)
        parameters: - original image
        return:     - Both Sobel filters concatenated
    """
    gray_image = cv2.cvtColor(data_in, cv2.COLOR_BGR2GRAY)
    
    one_layer_shape = [gray_image.shape[0], gray_image.shape[1], 1]
    gray_image = np.reshape(gray_image, one_layer_shape)
    
    # Sobel X
    # Output dtype = cv2.CV_64F, take absolute, convert to int
    sobelx64f = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize = 5)
    sobelx8u = np.uint8(np.absolute(sobelx64f))
    
    # Sobel Y
    # Output dtype = cv2.CV_64F, take absolute, convert to int
    sobelx64f = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize = 5)
    sobely8u = np.uint8(np.absolute(sobelx64f))
    
    # Concatenate Sobel filters to input image
    sobelx8u = np.reshape(sobelx8u, one_layer_shape)
    sobely8u = np.reshape(sobely8u, one_layer_shape)
    data_out = np.concatenate((sobelx8u, sobely8u), axis = 2)

    return data_out

def extract_Laplacian_filter(data_in):
    """ Function to run Laplacian filters
        parameters: - original image
        return:     - Laplacian filter
    """
    gray_image = cv2.cvtColor(data_in, cv2.COLOR_BGR2GRAY)
    
    one_layer_shape = [gray_image.shape[0], gray_image.shape[1], 1]
    gray_image = np.reshape(gray_image, one_layer_shape)
    
    # Laplacian
    # Output dtype = cv2.CV_64F, take absolute, convert to int
    laplacian64f = cv2.Laplacian(gray_image, cv2.CV_8U, ksize = 3)
    laplacian8u = np.uint8(laplacian64f)
   
    # Concatenate Laplacian filter to input image
    laplacian8u = np.reshape(laplacian8u, one_layer_shape)

    return laplacian8u

# Extract mean and var of each layer in the 3rd dimension 
def extract_features(img):
    res_Sobel = extract_Sobel_filter(img)
    # res_Laplacian = extract_Laplacian_filter(img)
    data_out = np.concatenate((img, res_Sobel), axis = 2)
    
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X

def features_augmentation(X):
    feat = np.zeros([X.shape[0], X.shape[1] + 1])
    for j in range(X.shape[0]):
        feat[j,:] = np.concatenate(([1],X[j,:]))
    return feat

def normalize(X):
    for i in range(X.shape[0]):
        d = data[i,:,:]
        d -= np.mean(d) 
        d /= np.linalg.norm(d) 

        # Update value
        X[i] = d 
        return X


# ***** Convert array of labels to an image *****

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def create_submission(y_pred, submission_filename, patch_size = 16, images_size = 608):
    n_patches = images_size // patch_size
    y_pred = np.reshape(y_pred, (-1, n_patches, n_patches))
    
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                for k in range(y_pred.shape[2]):
                    name = '{:03d}_{}_{},{}'.format(i+1, j * patch_size, k * patch_size, int(y_pred[i,j,k]))
                    f.write(name + '\n')
    