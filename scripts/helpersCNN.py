import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import matplotlib.image as mpimg
from sklearn.model_selection import KFold
from scipy import ndimage
import tensorflow as tf
from scipy import signal

# load an image from file path
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

# convert image from float to uint8
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# concatenate an image and its groundtruth
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

# split dataset into training and validation sets
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    num_row=len(y)
    indices=np.random.permutation(num_row)
    index_split=int(np.floor(ratio*num_row))
    index_tr=indices[: index_split]
    index_te=indices[index_split :]
    
    x_tr=x[index_tr]
    y_tr=y[index_tr]
    x_te=x[index_te]
    y_te=y[index_te]
    return x_tr, y_tr, x_te, y_te

# assign label to patch 
def value_to_class(v, foreground_threshold = 0.25):
    df = np.sum(v)
    if df > foreground_threshold:
        return [0,1]
    else:
        return [1,0]

# balance data i.e. returning set which classes contributions are 50/50
def balance_data(data, labels):
    c0 = 0
    c1 = 0
    for val in labels:
        if val[0] == 0:
            c0 += 1
        else:
            c1 += 1
    min_ = c0 if c0 < c1 else c1
    ind_datac0 = [i for i,j in enumerate(labels) if j[0] == 0]
    ind_datac1 = [i for i,j in enumerate(labels) if j[0] == 1]
    balanced_data = np.asarray([y for x in [data[ind_datac0[:min_]], data[ind_datac1[:min_]]] for y in x])
    balanced_labels = np.asarray([y for x in [labels[ind_datac0[:min_]], labels[ind_datac1[:min_]]] for y in x])
    return balanced_data, balanced_labels

# crop image into shape w x h or w*2*border x h*2*border if border parameter non zero and crops every # step pixels
def img_crop(im, w, h, border = 0, step = 16):
    """
    Return the patches list of an image.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    if border != 0:
        im_r = np.pad(im[:,:,0], ((border, border), (border, border)), 'reflect')
        im_g = np.pad(im[:,:,1], ((border, border), (border, border)), 'reflect')
        im_b = np.pad(im[:,:,2], ((border, border), (border, border)), 'reflect')
        im = np.dstack((im_r, im_g, im_b))
    for i in range(0,imgheight,step):
        for j in range(0,imgwidth,step):
            if is_2d:
                im_patch = im[j:j+w+2*border, i:i+h+2*border]
            else:
                im_patch = im[j:j+w+2*border, i:i+h+2*border, :]
            list_patches.append(im_patch)
    return list_patches

# rotate input images with angles submitted
def get_rotated_images(images, angles):
    rotated_images = [None]*(len(images)*len(angles))
    i = 0
    for angle in angles:
        for image in images:
            rotated_images[i] = ndimage.rotate(image, angle, mode='reflect', order=0, reshape=False)
            i += 1
    return rotated_images

# Convert array of labels to an image 
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

# create an image with img and its predicted img overlayed
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

# create submission file for kaggle
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
                  
# predict labels of test images
def predictions_on_test_images(img_patches_test, display_step = 1):
    kf = KFold(n_splits = 50)

    pred_list = []

    for i, [_, test_index] in enumerate(kf.split(img_patches_test)):
        if display_step:
            print("Fold ", i, " -- TEST:", test_index)
        X_test = img_patches_test[test_index]

        feed_dict = {x: X_test, keep_prob: 1.0}
        pred = predict.eval(feed_dict)

        pred_list.append(pred) 

    predictions = pred_list[0]
    for i in range(1, len(pred_list)):
        predictions = np.concatenate([predictions, pred_list[i]], axis = 0)
    return predictions