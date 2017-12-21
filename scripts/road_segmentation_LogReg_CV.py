import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

from sklearn import metrics, cross_validation
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


import time

try: 
    import cv2
except: 
    import pip
    pip.main(['install', 'opencv-python'])
    import cv2 
    
    


# Loaded a set of images
root_dir = "../training/"
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"

files = os.listdir(image_dir)

n = len(files)
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

print("Loading " + str(n) + " satellite + ground truth images")


seed = 3
np.random.seed(seed)
imgs = np.random.permutation(imgs)
gt_imgs = np.random.permutation(gt_imgs)


patch_size = 16

kf = KFold(n_splits = 10)
kf.get_n_splits(imgs)


# Cs = np.arange(1e4, 1e4 + 1, 10000) 
lambdas = np.logspace(-5, 0, 10)
Cs = [np.ceil(1/lambda_) for lambda_ in lambdas]

acc_threshold =[]

accuracy_train_C = []
f1_score_train_C = []
mse_train_C = []
rmse_train_C = []

accuracy_test_C = []
f1_score_test_C = []
mse_test_C = []
rmse_test_C = []

accuracy_test_C_pp = []
f1_score_test_C_pp = []
mse_test_C_pp = []
rmse_test_C_pp = []

start_time_total = time.time()
for C in Cs:
    start_time_C = time.time()
    print("C = {}".format(C))

    accuracy_train_CV = []
    accuracy_test_CV = []
    mse_train_CV = []
    rmse_train_CV = []
    
    f1_score_train_CV = []
    f1_score_test_CV = []
    mse_test_CV = []
    rmse_test_CV = []
    
    accuracy_test_CV_pp = []
    f1_score_test_CV_pp = []
    mse_test_CV_pp = []
    rmse_test_CV_pp = []

    for ind, [train_index, test_index] in enumerate(kf.split(imgs)):

        #
        # Split dataset for Cross Validation
        #
        print("\n{}-th CV".format(ind+1))

        X_train = [imgs[ind] for ind in train_index]
        X_test = [imgs[ind] for ind in test_index]

        y_train = [gt_imgs[ind] for ind in train_index]
        y_test = [gt_imgs[ind] for ind in test_index]

        #
        # Crop images, extract features, features augmentation and standardization
        # For both train & test datasets
        # 
        X_train = [img_crop(X_train[i], patch_size, patch_size, step = 8) for i in range(len(train_index))]
        y_train = [img_crop(y_train[i], patch_size, patch_size, step = 8) for i in range(len(train_index))]
        X_test = [img_crop(X_test[i], patch_size, patch_size) for i in range(len(test_index))]
        y_test = [img_crop(y_test[i], patch_size, patch_size) for i in range(len(test_index))]        

        X_train = np.asarray([X_train[i][j] for i in range(len(X_train)) for j in range(len(X_train[i]))])
        X_test = np.asarray([X_test[i][j] for i in range(len(X_test)) for j in range(len(X_test[i]))])
        y_train = np.asarray([y_train[i][j] for i in range(len(y_train)) for j in range(len(y_train[i]))])
        y_test = np.asarray([y_test[i][j] for i in range(len(y_test)) for j in range(len(y_test[i]))])

        y_train = np.asarray([value_to_class(np.mean(y_train[i])) for i in range(y_train.shape[0])])
        y_test = np.asarray([value_to_class(np.mean(y_test[i])) for i in range(y_test.shape[0])])

        # 
        # Balancing train data
        # 
        c0, c1 = 0, 0
        for i in range(len(y_train)):
            if y_train[i] == 0:
                c0 = c0 + 1
            else:
                c1 = c1 + 1
                
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(y_train) if j == 0]
        idx1 = [i for i, j in enumerate(y_train) if j == 1]
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        X_train = X_train[new_indices]
        y_train = y_train[new_indices]
        
        
        
        X_train = np.asarray([ extract_features(X_train[i]) for i in range(len(X_train))])
        X_train = features_augmentation(X_train)

        X_test = np.asarray([ extract_features(X_test[i]) for i in range(len(X_test))])
        X_test = features_augmentation(X_test)
        print("All data balanced and ready!")
        print(X_train.shape)

        
        # 
        # Run logistic regression 
        # 
        logreg = linear_model.LogisticRegression(C=C, class_weight="balanced")
        logreg.fit(X_train, y_train)
        z_train = logreg.predict(X_train)
        z_test = logreg.predict(X_test)

        # 
        # Compute f1 score & accuracy using sklearn functions
        # 
        f1_score_train = f1_score(y_train, z_train, average='micro')
        accuracy_score_train = accuracy_score(y_train, z_train)
        mse_train = mean_squared_error(y_train, z_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, z_train))
        
        f1_score_test = f1_score(y_test, z_test, average='macro')
        accuracy_score_test = accuracy_score(y_test, z_test)
        mse_test = mean_squared_error(y_test, z_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, z_test))
        
        print("Train f1_score : {}".format(f1_score_train))
        print("Test f1_score : {}".format(f1_score_test))
        
        
        # 
        # Post processing on test dataset
        # 
        # Reshape prediction as matrix for each image
        z_reshaped = []
        num_patch_total = len(z_test)
        num_patch_by_img = num_patch_total // len(test_index)
        
        for i in range(0, num_patch_total, num_patch_by_img):
            z_crt = z_test[i : i + num_patch_by_img]
            z_reshaped.append(np.reshape(z_crt, [400 // 16, 400 // 16]))

        # Run post process 
        for ind, label_img in enumerate(z_reshaped):
            label_img = postprocess(label_img)
            z_reshaped[ind] = np.reshape(label_img, [z_crt.shape[0]])

        # Convert list as array
        z_test_pp = np.concatenate( z_reshaped , axis = 0 )

        f1_score_test_pp = f1_score(y_test, z_test_pp, average='micro')
        accuracy_score_test_pp = accuracy_score(y_test, z_test_pp)
        mse_test_pp = mean_squared_error(y_test, z_test_pp)
        rmse_test_pp = np.sqrt(mean_squared_error(y_test, z_test_pp))
        
        print("Test f1_score post processed : {}\n".format(f1_score_test_pp))
        
        # 
        # Store accuracy for train, test and test+PP
        # 
        f1_score_train_CV.append(f1_score_train)
        accuracy_train_CV.append(accuracy_score_train)
        mse_train_CV.append(mse_train)
        rmse_train_CV.append(rmse_train)

        f1_score_test_CV.append(f1_score_test)
        accuracy_test_CV.append(accuracy_score_test)
        mse_test_CV.append(mse_test)
        rmse_test_CV.append(rmse_test)
        
        f1_score_test_CV_pp.append(f1_score_test_pp)
        accuracy_test_CV_pp.append(accuracy_score_test_pp)
        mse_test_CV_pp.append(mse_test_pp)
        rmse_test_CV_pp.append(rmse_test_pp)
    

    print("Average train F1-score: {}".format(np.mean(f1_score_train_CV)))
    
    print("Average test F1-score: {}".format(np.mean(f1_score_test_CV)))
    print("Variance test F1-score: {}".format(np.std(f1_score_test_CV)))
    print("Min test F1-score: {} // Max test F1-score: {}\n".format(np.min(f1_score_test_CV), np.max(f1_score_test_CV)))
    
    print("Average test F1-score PP: {}".format(np.mean(f1_score_test_CV_pp)))
    print("Variance test F1-score PP: {}".format(np.std(f1_score_test_CV_pp)))
    print("Min test F1-score PP: {} // Max test F1-score PP: {}\n".format(np.min(f1_score_test_CV_pp), np.max(f1_score_test_CV_pp)))
    
    accuracy_train_C.append(np.mean(accuracy_train_CV))
    f1_score_train_C.append(np.mean(f1_score_train_CV))
    mse_train_C.append(np.mean(mse_train_CV))
    rmse_train_C.append(np.mean(rmse_train_CV))

    accuracy_test_C.append(np.mean(accuracy_test_CV))
    f1_score_test_C.append(np.mean(f1_score_test_CV))
    mse_test_C.append(np.mean(mse_test_CV))
    rmse_test_C.append(np.mean(rmse_test_CV))
    
    accuracy_test_C_pp.append(np.mean(accuracy_test_CV_pp))
    f1_score_test_C_pp.append(np.mean(f1_score_test_CV_pp))
    mse_test_C_pp.append(np.mean(mse_test_CV_pp))
    rmse_test_C_pp.append(np.mean(rmse_test_CV_pp))
    
    
    print("Time for C = %s --- %s seconds --- \n\n" % (C, time.time() - start_time_C))
print("Total time --- %s seconds --- \n\n" % (time.time() - start_time_total))
        





