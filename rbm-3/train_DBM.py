#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import datetime
import argparse
import re
import glob
from obj.DBM import DBM
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize

################################
# train DBM from input data
################################

numpy_path = "gait_normalized.npy"

def trainDBM(data, learning_rate, k1, k2, epochs, batch_size, dims):
    # import data
    print("importing training data")
    if data == "fashion_mnist":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, _), (_,_) = fashion_mnist.load_data()
    elif data == "mnist":
        mnist = tf.keras.datasets.mnist
        (x_train, _), (_,_) = mnist.load_data()
    elif data == "faces":
        x_train = [resize(mpimg.imread(file),(28,28)) for file in glob.glob("data/faces/*")]
        x_train = np.asarray(x_train)
        # make images sparse for easier distinctions
        for img in x_train:
            img[img < np.mean(img)+0.5*np.std(img)] = 0
    elif data == "gait":
        x_train = np.load(numpy_path) # load
    else:
        raise NameError("unknown data type: %s" % data)
    if data == "mnist" or data == "fashion_mnist":
        x_train = x_train/255.0
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    elif data == "faces":
        # auto conversion to probabilities in earlier step
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    # create log directory
    current_time = getCurrentTime()+"_"+re.sub(",","_",dims)+"_"+data+"_dbm"
    os.makedirs("pickles/"+current_time)
    # parse string input into integer list
    dims = [int(el) for el in dims.split(",")]
    dbm = DBM(dims, learning_rate, k1, k2, epochs, batch_size)
    dbm.train_PCD(x_train)
    # dump dbm pickle
    f = open("pickles/"+current_time+"/dbm.pickle", "wb")
    pickle.dump(dbm, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

####################################
# main command call
####################################

if __name__ == "__main__":
    data = 'gait'
    learning_rate = .01
    k1 = 1
    k2 = 5
    epochs = 1
    batch_size = 160
    dimensions = '6,3,6,3'
    trainDBM(data,learning_rate,k1,k2,epochs,batch_size,dimensions)
