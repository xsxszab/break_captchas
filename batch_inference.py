import os
import string
import warnings
import itertools
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dependencies from keras,
# for any question please refer to keras's doc
# https://keras.io/
from keras.models import Input, Model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, Dense, GRU, merge, Dropout, Lambda
from keras import backend as K

from utils.util_func import processImg # image processing tool

warnings.filterwarnings('ignore') # ignore warnings


parser = argparse.ArgumentParser()

parser.description = 'model training script'
parser.add_argument('-b', '--batchsize', help='batch size', type=int, default=128)
parser.add_argument('-l', '--height', help='image height', type=int, default=40)
parser.add_argument('-w', '--width', help='image width', type=int, default=120)
parser.add_argument('-imgdir', '--img_dir', help='image dir', type=str, default='./test/')
parser.add_argument('-outdir', '--output_dir', help='output dir', type=str, default='./')
parser.add_argument('-modeldir', '--model_dir', help='model save dir', type=str, default='./model/')
args = parser.parse_args()


img_dir = args.img_dir
model_save_dir = args.model_dir
output_dir = args.output_dir
batch_size = args.batchsize
height = args.height
width = args.width
rnn_size = 256
characters = string.digits + string.ascii_lowercase + string.ascii_uppercase # table of symbols appear in the dataset
n_class = len(characters)+1
n_len = 4 # length of words in each image

imgs = []
pred_labels = [] # store predicted labels

l = os.listdir(img_dir)
l.sort(key= lambda x: int(x[:-4])) # make sure the sequence of images is not shuffled
for name in l:
    processImg(img_path=img_dir, name=name, imgs=imgs, method=['memory'])


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

input_tensor = Input((width, height, 1)) # define input shape, if input if rgb format, switch the 3rd dimension to 3 
x = input_tensor
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

model.load_weights(model_save_dir+'model.h5') # load model

for img in imgs:
    img = img.T
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    y_pred = base_model.predict(img)
    y_pred = y_pred[:,2:,:]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([characters[x] for x in out[0]])
    pred_labels.append(out)

data = pd.DataFrame({'ID':l, 'label':pred_labels})
data.to_csv(output_dir+'pred.csv', index=False)
