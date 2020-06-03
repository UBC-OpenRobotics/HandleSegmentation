#!/usr/bin/python
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
import json
import argparse

import tensorflow as tf
from tensorflow.keras import datasets, layers, Model, utils, optimizers, models, metrics
from tensorflow.keras.layers import Conv2D,concatenate, Activation,UpSampling2D, BatchNormalization, MaxPooling2D, Dropout,Conv2DTranspose

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = UpSampling2D((2,2), interpolation='bilinear')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    #u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = UpSampling2D((2,2), interpolation='bilinear')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    #u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = UpSampling2D((2,2), interpolation='bilinear')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    #u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = UpSampling2D((2,2), interpolation='bilinear')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-r','--root_path', type=str, required=True, help='path to HandleSegmentation root folder')
    args = vars(ap.parse_args())

    base_path = args['root_path']
    #Read configuration file
    with open(os.path.join(base_path,'config.json')) as f:
        config = json.load(f)

    dataset_path = config['dataset_path']
    w = config['input_w']
    h = config['input_h']

    train_img_dir = config['train_img_dir']
    train_mask_dir = config['train_mask_dir']
    dev_img_dir = config['dev_img_dir']
    dev_mask_dir = config['dev_mask_dir']

    model_path = config['model_path']
    epochs = config['epochs']
    batch_size = config['batch_size']
    figure_path = config['figure_path']


    #Get absolute paths
    dataset_path = os.path.join(base_path, dataset_path)
    train_img_dir = os.path.join(base_path, train_img_dir)
    train_mask_dir = os.path.join(base_path, train_mask_dir)
    dev_img_dir = os.path.join(base_path, dev_img_dir)
    dev_mask_dir = os.path.join(base_path, dev_mask_dir)
    model_path = os.path.join(base_path, model_path)
    figure_path = os.path.join(base_path, figure_path)


    X = []
    y = []

    X_dev = []
    y_dev = []

    #Kernels for morphological close and dilate
    kernel_c = np.ones((10,10),np.uint8)
    kernel_d = np.ones((5,5),np.uint8)

    #Get image extension
    i=0
    dir_size = len(os.listdir(train_img_dir))
    sample_img = os.listdir(train_img_dir)[i]
    img_ext = sample_img.split('.')[-1]
    
    while img_ext not in ['jpg','png'] and i < dir_size:
        sample_img = os.listdir(train_img_dir)[i]
        img_ext = sample_img.split('.')[-1]
        i+=1
    if i==dir_size:
        print('Error, could not find .jpg or .png images in %s'%(train_img_dir))
        exit()

    for mask_name in tqdm(os.listdir(train_mask_dir), desc="Collecting Training Data"):
        #Read in Mask
        mask_path = os.path.join(train_mask_dir, mask_name)
        mask = cv2.imread(mask_path,0)
        
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        #Close gaps in mask and dilate
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
        mask = cv2.dilate(mask, kernel_d,iterations = 1)
        
        #Read in Image
        img_name = mask_name.split('.')[0]+'.'+img_ext
        img_path = os.path.join(train_img_dir, img_name)
        img = cv2.imread(img_path, 0)
        assert img.shape == (w,h), 'Image shape is not (%i,%i)' %(w,h)
        
        #Normalize and save
        X.append(img/255.0)
        y.append(mask/255.0)

    for mask_name in tqdm(os.listdir(dev_mask_dir), desc="Collecting Dev Data"):
        #Read in Mask
        mask_path = os.path.join(dev_mask_dir, mask_name)
        mask = cv2.imread(mask_path,0)
        
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        #Close gaps in mask and dilate
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
        mask = cv2.dilate(mask, kernel_d,iterations = 1)
        
        #Read in Image
        img_name = mask_name.split('.')[0]+'.'+img_ext
        img_path = os.path.join(dev_img_dir, img_name)
        
        img = cv2.imread(img_path, 0)
        assert img.shape == (w,h), 'Image shape is not (%i,%i)' %(w,h)
        
        #Normalize and save
        X_dev.append(img/255.0)
        y_dev.append(mask/255.0)

    #Save X and y as np arrays
    X = np.asarray(X)
    y = np.asarray(y)
    X_dev = np.asarray(X_dev)
    y_dev = np.asarray(y_dev)

    print('Shape of X: %s\nShape of y: %s\n\n'%(X.shape, y.shape))
    print('Shape of X_dev: %s\nShape of y_dev: %s'%(X_dev.shape, y_dev.shape))

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,9)
    fig.set_facecolor('w')

    ax[0].imshow(X[0])
    ax[0].set_title('First image of X')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].imshow(y[0])
    ax[1].set_title('First Mask of y')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    plt.savefig(os.path.join(figure_path, 'dataset_first_image.png'))

    #Before feeding into model, need to expand dims
    if len(X.shape) < 4:
        X = np.expand_dims(X, axis=3)
    if len(y.shape) < 4:
        y = np.expand_dims(y, axis=3)

    assert X.shape == (len(X),w,h,1),'Incorrect dimensions for X, expected %s but got %s' %(str([len(X),w,h,1]), X.shape)
    assert y.shape == (len(y),w,h,1), 'Incorrect dimensions for y, expected %s but got %s' %(str([len(y),w,h,1]), y.shape)

    #Create and compile model
    inputs = layers.Input(X[0].shape)
    model = get_unet(inputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dice_loss, metrics=[dice_loss,'accuracy'])


    #Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    idx = np.random.permutation(len(X_dev))
    X_dev = X[idx]
    y_dev = y[idx]

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_dev,y_dev))

    model.save(model_path)

    print('Model saved')

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)
    fig.set_facecolor('w')
    ax.plot(history.history['acc'], label='accuracy')
    ax.plot(history.history['val_acc'], label='validation_accuracy')
    ax.plot(history.history['dice_loss'], label='dice_loss')
    ax.plot(history.history['val_dice_loss'], label='validation_dice_loss')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(figure_path,'history.png'))