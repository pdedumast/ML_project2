
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf


# **Imports**

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from helpersCNN import *
from scipy import ndimage
import random
import sklearn as sk
from copy import copy, deepcopy

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


# **Loading set of training images and their corresponding groundtruth images**

# Loaded a set of images
current_dir = './'
root_dir = current_dir + "data/training/"

# image data
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(100, len(files)) # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
print(files[0])

# image labels
gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
print(files[0])


# **Rotate training images and their groundtruth related images with 12 randomly chosen angles, 3 for each interval: [0,90], [90,180], [180, 270], [270, 360]**

angles = [23,121,236,327]

rotated_imgs = get_rotated_images(imgs, angles)
gt_rotated_imgs = get_rotated_images(gt_imgs, angles)


# **Concatenate original images and the rotated images**

imgs = imgs + rotated_imgs 
gt_imgs = gt_imgs + gt_rotated_imgs 


# **Display first original training image and one of its rotation**

#figure = plt.figure(figsize=(15, 15))
#plt.subplot(121)
#plt.imshow(imgs[0])
#plt.subplot(122)
#plt.imshow(imgs[100])
#plt.savefig('rotation.png')


# **Extract 16x16 patches from all training images and their corresponding groundtruth images**
# Training patches become 48x48 patches because of 16 pixels border 

# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels
border = 16
n = len(imgs)

img_patches = [img_crop(imgs[i], patch_size, patch_size, border, step = 16) for i in range(n)]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size, step = 16) for i in range(n)]

print(len(img_patches))
print(len(img_patches[0]))
print(img_patches[0][0].shape)

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

print(img_patches.shape)
print(img_patches[0].shape)


# **Standardize training images**

mean_r = np.mean(img_patches[:,:,:,0])
mean_g = np.mean(img_patches[:,:,:,1])
mean_b = np.mean(img_patches[:,:,:,2])

imgs_r = img_patches[:,:,:,0]
imgs_g = img_patches[:,:,:,1]
imgs_b = img_patches[:,:,:,2]
print('patches duplicated')
std_imgs_r = imgs_r - mean_r 
std_imgs_g = imgs_g - mean_g 
std_imgs_b = imgs_b - mean_b

std_r = np.std(imgs_r)
std_g = np.std(imgs_g)
std_b = np.std(imgs_b)

if std_r > 0: 
    std_imgs_r /= std_r
if std_g > 0: 
    std_imgs_g /= std_g
if std_b > 0: 
    std_imgs_b /= std_b

imgs_patches = np.stack((std_imgs_r, std_imgs_g, std_imgs_b), axis=3)


# **Assigning labels to groundtruth patches to determine wether they are road or background**

train_labels = np.asarray([value_to_class(np.mean(gt_patches[i]), 0.25) for i in range(len(gt_patches))])


# **Balancing the data**

img_patches, train_labels = balance_data(img_patches, train_labels)


# **Splitting data into training and validation sets**
# Rate set to 0.9 for optimal training to have more input data

train_data, train_labels, eval_data, eval_labels = split_data(img_patches, train_labels, 0.9, seed)


# **Define CNN architecture**

#inputs: num_training_imagesx48x48x3 images
#ouputs: num_training_imagesx2 vector holding probabilities for each class

x = tf.placeholder(tf.float32, [None, 48,48,3])
y_ = tf.placeholder(tf.float32, [None, 2])


#initialization of weight variable function
def weight_variable(shape):
    n = np.multiply.reduce(shape[:-1])
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)

#initialization of bias variable function
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#convolution function of input x and filter W with stride one 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides =[1, 1, 1, 1], padding = 'SAME')

#pooling function of size 2x2 and stride 2 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding ='SAME')

#first convolutional layer with filter size = [5x5x3] 
#and number of feature maps = 32
W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])


#first ReLU 
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)


#first pooling
h_pool1 = max_pool_2x2(h_conv1)


#second convolutional layer with filter size = [5x5x32] 
#and number of feature maps = 64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])


#second ReLU 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)


#second pooling
h_pool2 = max_pool_2x2(h_conv2)


#fully connected layer
W_fcl = weight_variable([12*12*64, 1024])
b_fcl = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])

h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)


keep_prob = tf.placeholder('float')
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob,seed=seed)


#classification of images using softmax activation in the fc layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2)+b_fc2)


#define training steps loss and optimizer as well as 
#accuracy and prediction tools
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
l2_regularization_penalty = 0.01
l2_loss = l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) + 
                                       tf.nn.l2_loss(W_conv2) +
                                       tf.nn.l2_loss(W_fcl) +
                                       tf.nn.l2_loss(W_fc2))
loss = tf.add(cross_entropy, l2_loss, name='loss')
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)


# **Training on whole dataset is highly computationally costly so we use batches to train the CNN at each iteration**

epochs_completed = 0
index_in_epoch = 0
num_samples = train_data.shape[0]

def next_batch(batch_size,train_data,train_labels,index_in_epoch,epochs_completed,seed):
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_samples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        np.random.seed(seed)
        perm = np.random.permutation(num_samples)
        train_data = train_data[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return train_data[start:end], train_labels[start:end], epochs_completed, index_in_epoch


# **Initialize Variables of the CNN**

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)


#define parameters for training
TRAINING_ITERATIONS = 10000        
DROPOUT = 0.5
VALIDATION_SIZE = eval_labels.shape[0]
BATCH_SIZE = 100

training_accuracies = []
training_f1scores = []
validation_accuracies = []
validation_f1scores = []
display_steps = []
display_step = 1

for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys, epochs_completed, index_in_epoch = next_batch(BATCH_SIZE,train_data,train_labels,index_in_epoch,epochs_completed,seed)  
    # display accuracy and F1-score for training and validation
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        training_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
        training_predictions = predict.eval(feed_dict={x:batch_xs, keep_prob: 1.0})
        f1_score_training = sk.metrics.f1_score(np.argmax(batch_ys,axis=1), training_predictions)
        training_f1scores.append(f1_score_training)

        if(VALIDATION_SIZE):
            validation_predictions = predict.eval(feed_dict={x: eval_data[0:BATCH_SIZE], keep_prob: 1.0})
            validation_accuracy = accuracy.eval(feed_dict={x: eval_data[0:BATCH_SIZE], y_: eval_labels[0:BATCH_SIZE], keep_prob: 1.0})
            f1_score_validation = sk.metrics.f1_score(np.argmax(eval_labels[0:BATCH_SIZE],axis=1), validation_predictions)
            validation_f1scores.append(f1_score_validation)
            print('training_accuracy = %.2f // validation_accuracy = %.2f // STEP = %d'%(training_accuracy, validation_accuracy, i))
            print('training_f1score = %.2f // validation_f1score = %.2f // STEP = %d'%(f1_score_training, f1_score_validation, i))
            validation_accuracies.append(validation_accuracy)
            if validation_accuracy > 0.93:
                break
        else:
            print('training_accuracy = %.4f // STEP = %d'%(training_accuracy, i))
            print('training_f1score = %.2f // STEP = %d'%(f1_score_training, i))
        training_accuracies.append(training_accuracy)
        
        display_steps.append(i)
        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train CNN
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


#if(VALIDATION_SIZE):
#    plt.plot(display_steps,validation_f1scores,'b',label='Validation F1-score')
#plt.plot(display_steps,training_f1scores,'r',label='Training F1-score')
#plt.ylabel('f1-score')
#plt.xlabel('iterations')
#plt.title('Training and validation f1-scores function of # of iterations')
#plt.legend()
#plt.show()


# **Loading test images**

# Data to evaluate
root_testdir = "data/test_set_images"
test_names = os.listdir(root_testdir)

prefixes = ('.')
for dir_ in test_names[:]:
    if dir_.startswith(prefixes):
        test_names.remove(dir_)
        
num_test = len(test_names)

#get data permutation
order = [int(test_names[i].split("_")[1]) for i in range(num_test)]
p = np.argsort(order)

imgs_test = [load_image(os.path.join(root_testdir, test_names[i], test_names[i]) + ".png") for i in range(num_test)]
#order data
imgs_test = [imgs_test[i] for i in p]

#get patches of test images
img_patches_test = [img_crop(imgs_test[i], patch_size, patch_size, border, step = 16) for i in range(num_test)]
# Linearize list of patches
img_patches_test = np.asarray([img_patches_test[i][j] for i in range(len(img_patches_test)) for j in range(len(img_patches_test[i]))])


# **Standardize test patches**

mean_r_te = np.mean(img_patches_test[:,:,:,0])
mean_g_te = np.mean(img_patches_test[:,:,:,1])
mean_b_te = np.mean(img_patches_test[:,:,:,2])

imgs_r_te = img_patches_test[:,:,:,0]
imgs_g_te = img_patches_test[:,:,:,1]
imgs_b_te = img_patches_test[:,:,:,2]

std_imgs_r_te = imgs_r_te - mean_r_te
std_imgs_g_te = imgs_g_te - mean_g_te 
std_imgs_b_te = imgs_b_te - mean_b_te

std_r_te = np.std(imgs_r_te)
std_g_te = np.std(imgs_g_te)
std_b_te = np.std(imgs_b_te)

if std_r_te > 0: 
    std_imgs_r_te /= std_r_te
if std_g_te > 0: 
    std_imgs_g_te /= std_g_te
if std_b_te > 0: 
    std_imgs_b_te /= std_b_te
    
imgs_patches_test = np.stack((std_imgs_r_te, std_imgs_g_te, std_imgs_b_te), axis=3)


# **Get predictions of test patches**

from sklearn.model_selection import KFold
kf = KFold(n_splits = 50)

pred_list = []

#predicting patches in batches
for i, [_, test_index] in enumerate(kf.split(img_patches_test)):
    print("Fold ", i, " -- TEST:", test_index)
    X_test = img_patches_test[test_index]
    
    feed_dict = {x: X_test, keep_prob: 1.0}
    pred = predict.eval(feed_dict)
    
    pred_list.append(pred) 

#building predictions vector
predictions = pred_list[0]
for i in range(1, len(pred_list)):
    predictions = np.concatenate([predictions, pred_list[i]], axis = 0)


# **Create submission file for kaggle**

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

create_submission(predictions, "submissionCNN.csv")


# **Display prediction overlay on test image**

#choose image index
#img_idx = 0

#predict labels for image
#feed_dict = {x: img_patches_test[(img_idx)*38*38:(img_idx+1)*38*38], keep_prob: 1.0}
#Zi = predict.eval(feed_dict)

# Display prediction as an image
#patch_size = 16
#w = imgs_test[img_idx].shape[0]
#h = imgs_test[img_idx].shape[1]
#predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
#fig1 = plt.figure(figsize=(15, 15)) 
#new_img = make_img_overlay(imgs_test[img_idx], predicted_im)

#plt.subplot(121)
#plt.imshow(new_img)

#plt.subplot(122)
#plt.imshow(predicted_im)

#plt.savefig('results.png')

