import pyBigWig
import argparse
import os
import sys
import numpy as np
import re
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
from datetime import datetime
import unet
import random
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.45 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size=2**13
num_channel=5

num_sample=10000
batch_size=10

path1='../../data/bigwig/'
path2='../../data/dropdown_bigwig/'
path3='../../data/dna_bigwig/'

chr_all=[]
for i in range(1,20):
    chr_all.append(str(i))

chr_all.append('X')

# GRCm38
num_bp=[195471971,182113224,160039680,156508116,151834684, \
    149736546,145441459,129401213,124595110,130694993, \
    122082543,120129022,120421639,124902244,104043685, \
    98207768,94987271,90702639,61431566,171031299]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]


## sample index for chunks ###########
index_chr=np.array([])
freq=np.rint(np.array(num_bp)/sum(num_bp)*1000).astype('int')
for i in np.arange(len(chr_all)):
    index_chr = np.hstack((index_chr, np.array([chr_all[i]] * freq[i])))
np.random.shuffle(index_chr)
#############################################

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-f', '--fold', default='0', type=int, help='fold for test partition')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    args = parser.parse_args()
    return args

args=get_args()
fold_partition = args.fold
seed_partition = args.seed

name_model='weights_fold' + str(fold_partition) + '_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-3,num_class=1,num_channel=num_channel,size=size)
#model.load_weights(name_model)
model.summary()

## train-vali-test partition
id_all=np.loadtxt('./partition/id_all.txt', dtype='str')
id_test=np.loadtxt('./partition/id_test' + str(fold_partition) + '.txt', dtype='str')
id_tv = []
for the_id in id_all:
    if the_id not in id_test:
        id_tv.append(the_id)

id_tv=np.array(id_tv)

np.random.seed(seed_partition) #TODO
np.random.shuffle(id_tv)
ratio=[0.75,0.25]
num = int(len(id_tv)*ratio[0])
id_train = id_tv[:num]
id_vali = id_tv[num:]

print('id_train:', id_train)
print('id_vali:', id_vali)
print('id_test:', id_test)
#############################################

# open bigwig
dict_label={}
dict_feature={}
for the_id in id_tv:
    dict_label[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_feature[the_id]=pyBigWig.open(path2 + the_id + '.bigwig')
list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path3 + the_id + '.bigwig')

############

##### augmentation parameters ######
if_time=False
max_scale=1.15
min_scale=1
if_mag=True
#if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=False
####################################

def generate_data(batch_size, if_train):
    i=0
    j=0
    while True:
        b = 0
        image_batch = []
        label_batch = []

        while b < batch_size:
            if (if_train==1):
                list_id = id_train.copy()
            else:
                list_id = id_vali.copy()

            if i == len(index_chr):
                i=0
                np.random.shuffle(index_chr)

            the_chr=index_chr[i]

            # random ids
            id_target = list_id[np.random.randint(0, len(list_id), 1)[0]]
            # random segments
            start=np.random.randint(0, chr_len[the_chr] - size, 1)[0]
            end=start + size

            label = np.array(dict_label[id_target].values(the_chr, start, end)).reshape((1,-1))

            image = np.zeros((num_channel, size))
            num=0
            for k in np.arange(len(list_dna)):
                the_id=list_dna[k]
                image[num,:] = dict_dna[the_id].values(the_chr, start, end)
                num+=1
            image[num,:] = np.array(dict_feature[id_target].values(the_chr, start, end))

            # augmentation
            if (if_train==1) and (if_mag):
                rrr=random.random()
                rrr_mag=rrr*(max_mag-min_mag)+min_mag
                image[4,:]=image[4,:]*rrr_mag                               

            image_batch.append(image.T)
            label_batch.append(label.T)

            b += 1
            i += 1

        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
        yield image_batch, label_batch


callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(batch_size,True),
    steps_per_epoch=int(num_sample // batch_size), nb_epoch=5,
    validation_data=generate_data(batch_size,False),
    validation_steps=int(num_sample // batch_size),callbacks=callbacks,verbose=1)

for the_id in list_dna:
    dict_dna[the_id].close()
for the_id in id_tv:
    dict_label[the_id].close()
    dict_feature[the_id].close()



