
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import matplotlib.image as mpimg
from sklearn.preprocessing import PolynomialFeatures


import cv2
from scipy import signal



############################################
#                                          #
#      Functions for Image Management      #
#                                          #
############################################


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

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


def img_crop(im, w, h,step = 16):
    """
    Return the patches list of an image.’
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,step):
        for j in range(0,imgwidth,step):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def get_transformed_images(imgs):
    """
    From a list of images, constructs and returns
    a list of the 4 rotated images plus 1 fliped image of the input ones.
    """
    rotations = [0, 90, 180, 270]
    transformed_images = []
    for img in imgs:
        # Let's rotate images
        for rotation in rotations:
            # Check if it is a rgb or a b&w picture
            if(len(img.shape)==3):
                rows, cols, _ = img.shape
            else:
                rows, cols = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))
            transformed_images.append(dst)
            # Let's flip the image
            transformed_images.append(cv2.flip(dst.copy(), 0))
    
    return transformed_images


<<<<<<< HEAD
def img_crop(im, w, h,step = 16):
    """
    Return the patches list of an image.’
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,step):
        for j in range(0,imgwidth,step):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def polynomial_augmentation(X,polynomial_degree= 4):
        """
        Fit the dataset using a polynomial augmentation.
        By default the augmentation degree is 4.
        """
        polynomial = PolynomialFeatures(polynomial_degree)
        return polynomial.fit_transform(X)


# ***** Features extraction *****
=======
###############################################
#                                             #
#      Functions for Features Extraction      #
#                                             #
###############################################
>>>>>>> 4de912ab6c3b193508d61b17ada95a869b04a3f1

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


def extract_features(img):
    """ 
    Add Layer for Sobel Filters in 3rd dim
    return vector of mean and var of each layer in the 3rd dimension

    """
    res_Sobel = extract_Sobel_filter(img)
    # res_Laplacian = extract_Laplacian_filter(img)
    data_out = np.concatenate((img, res_Sobel), axis = 2)
    img = data_out

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
def extract_img_features(filename, patch_size = 16):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)

    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X


def polynomial_augmentation(X, polynomial_degree = 2):
        """
        Fit the dataset using a polynomial augmentation.
        By default the augmentation degree is 3.
        """
        polynomial = PolynomialFeatures(polynomial_degree)
        return polynomial.fit_transform(X)

def features_augmentation(X):
    X = polynomial_augmentation(X)
    feat = np.zeros([X.shape[0], X.shape[1] + 1])
    
    for j in range(X.shape[0]):
        feat[j,:] = np.concatenate(([1],X[j,:]))
    return feat

def normalize(X):
    for i in range(X.shape[0]):
        d = X[i,:,:]
        d -= np.mean(d)
        d /= np.linalg.norm(d)

        # Update value
        X[i] = d
        return X

def demean_images(imgs):
    ''' Demean the input images'''
    mean_vec = np.zeros((3,1))
    for img in imgs:
        is_2d = len(img.shape) < 3
        if(is_2d):
            img -= np.mean(img)
        else:
            mean_vec[0] = np.mean(img[:,:,0])
            img[:,:,0] -= np.mean(img[:,:,0])

            mean_vec[1] = np.mean(img[:,:,1])
            img[:,:,1] -= np.mean(img[:,:,1])

            mean_vec[2] = np.mean(img[:,:,2])
            img[:,:,2] -= np.mean(img[:,:,2])

    demean_imgs = imgs
    return demean_imgs, mean_vec

def normalize_images(imgs):
    ''' Normalize the input images'''
    std_vec = np.zeros((3,1))

    for img in imgs:
        is_2d = len(img.shape) < 3
        if(is_2d):
            img /= np.std(img)
        else:
            std_vec[0] = np.std(img[:,:,0])
            img[:,:,0] /= np.std(img[:,:,0])

            std_vec[1] = np.std(img[:,:,1])
            img[:,:,1] /= np.std(img[:,:,1])

            std_vec[2] = np.std(img[:,:,2])
            img[:,:,2] /= np.std(img[:,:,2])

    normalized_imgs = imgs
    return normalized_imgs, std_vec

def standardize(imgs):
    ''' Demean and normalize the input images'''
    demean_imgs, mean_vec = demean_images(imgs)
    normalized_imgs, std_vec = normalize_images(demean_imgs)

    standardized_imgs = normalized_imgs
    return standardized_imgs, mean_vec, std_vec

def standardize_data(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def postprocess(img):
    """ Postprocessing of the predictions
    Modify isolated patchs """
    [dim_x, dim_y] = img.shape
    kernel = np.ones((3,3),np.float32)
    kernel[1,1] = 0

    filtered_img = signal.convolve2d(img, kernel)
    postprocess_img = img

    for i in range(0, dim_y):
        for j in range(0, dim_x):
            if img[i,j] == 1:
                if filtered_img[i,j] < 2 :
                    # If a patch is predicted as road,
                    # but less than 2 neighbors are also predicted road :
                    # Then we consider the patch not to be road
                    postprocess_img[i,j] = 0

            #elif img[i,j] == 0:
                    # If a patch is predicted as NOT road,
                    # but more than 7 neighbors are predicted road :
                    # Then we consider the patch to be road
                #if filtered_img[i,j] >= 7 :
                   # postprocess_img[i,j] = 1

    return postprocess_img



###############################################
#                                             #
#           Functions for Submission          #
#                                             #
###############################################

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
            img = y_pred[i]
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    name = '{:03d}_{}_{},{}'.format(i + 1, j * patch_size, k * patch_size, int(img[j,k]))
                    f.write(name + '\n')
