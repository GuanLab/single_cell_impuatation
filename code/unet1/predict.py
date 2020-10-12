#!/usr/bin/env python
import pyBigWig
import argparse
import os
import sys
import logging
import numpy as np
import re
import time
import scipy.io
import glob
import unet
import tensorflow as tf
import keras
from keras import backend as K
import scipy
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.13 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

###### PARAMETER ###############

size=2**13
num_channel=5
size_edge=int(100) # chunk edges to be excluded
batch=100

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

# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-m', '--model', default='weights_fold0_seed0.h5', type=str,
        help='model name')
    parser.add_argument('-f', '--fold', default='0', type=int, help='fold for test partition')
#    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition') #TODO
#    parser.add_argument('-t', '--test', default='AAAGATGGTCATATGC-1', nargs='+', type=str,
#        help='test cell type')
    args = parser.parse_args()
    return args

args=get_args()

print(args)
name_model=args.model
fold_partition=args.fold

model = unet.get_unet(the_lr=1e-3,num_class=1,num_channel=num_channel,size=size)
model.load_weights(name_model)
#model.summary()

## train-vali-test partition
id_test=np.loadtxt('./partition/id_test' + str(fold_partition) + '.txt', dtype='str')
id_test=np.sort(id_test)
print('id_test:', id_test)

# open bigwig
dict_label={}
dict_feature={}
for the_id in id_test:
    dict_label[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_feature[the_id]=pyBigWig.open(path2 + the_id + '.bigwig')
list_dna=['A','C','G','T']
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path3 + the_id + '.bigwig')

############

score_file=open('score_fold' + str(fold_partition) + '.txt','w')
score_file.write('id\tmse\tpear\n')

mse_all=[]
pear_all=[]

path_pred = './pred/'
os.system('mkdir -p ' + path_pred)

for id_target in id_test:
    print(id_target)

    bw_output = pyBigWig.open(path_pred + 'pred_' + id_target + '_fold' + str(fold_partition) + '.bigwig','w')
    bw_output.addHeader(list(zip(chr_all , num_bp)), maxZooms=0)

    #list_chr = chr_all
    list_chr = ['19'] #TODO
    for the_chr in list_chr:
        print('chr' + the_chr)
        pred_final=np.zeros(chr_len[the_chr])
        count_final=np.zeros(chr_len[the_chr])

        for phase in np.arange(0,1,0.5):
            i = 0 + int(size * phase)
            print('phase%.1f' % phase)
            if (i!=0): # shift the end as well
                d1 = chr_len[the_chr] - size + int(size * phase)
            else:
                d1 = chr_len[the_chr]

            progress_old = 0
            while (i<d1):
                # add progress bar
                progress = int(i/chr_len[the_chr]*100/2)
                if progress != progress_old:
                    print("[%-50s] %d%%" % ('='*progress, progress*2))
                    progress_old = progress
#                print(i, i/chr_len[the_chr])
                start = i
                end = i + size*batch
                if (end>d1):
                    end = d1
                    start = d1 - size*batch
                image = np.zeros((num_channel, size*batch))
                # dna
                num=0
                for j in np.arange(len(list_dna)):
                    the_id=list_dna[j]
                    image[num,:] = dict_dna[the_id].values(the_chr,start,end)
                    num+=1
                # feature & diff
                image[num,:] = np.array(dict_feature[id_target].values(the_chr, start, end))

                ## make predictions ################
                input_pred=np.reshape(image.T,(batch,size,num_channel))
                output1 = model.predict(input_pred)
                output1 = np.reshape(output1,(size*batch, 1)).T
                output_new=output1.flatten()

                i_batch=0
                while (i_batch<batch):
                    i_start = start + i_batch*size
                    i_end = i_start + size
                    if (i_start==0):
                        start_new = i_start
                        end_new = i_end - size_edge
                        start_tmp = 0 + i_batch*size
                        end_tmp = size - size_edge + i_batch*size
                    elif (i_end==d1):
                        start_new = i_start + size_edge
                        end_new = i_end
                        start_tmp = size_edge + i_batch*size
                        end_tmp = size + i_batch*size
                    else:
                        start_new = i_start + size_edge
                        end_new = i_end - size_edge
                        start_tmp = size_edge + i_batch*size
                        end_tmp = size - size_edge + i_batch*size
                    pred_final[start_new:end_new] += output_new[start_tmp:end_tmp]
                    count_final[start_new:end_new] += 1
                    i_batch += 1

                i=i+int(size*batch)

        del output1
        del output_new
        del image
        del input_pred

        ######################################################################
        
        ## scoring 
        pred_final=np.divide(pred_final,count_final)
        gt = np.array(dict_label[id_target].values(the_chr,0,chr_len[the_chr]))
        the_mse = mse(gt, pred_final)
        the_pear = np.corrcoef(gt, pred_final)[0,1]
        score_file.write('%s\t%.5f\t%.5f\n' % (id_target, the_mse, the_pear))
        score_file.flush()
        print(id_target, the_mse, the_pear)
        mse_all.append(the_mse)
        pear_all.append(the_pear)

        ## save bigwig
        x=pred_final
        # pad two zeroes
        z=np.concatenate(([0],x,[0]))
        # find boundary
        starts=np.where(np.diff(z)!=0)[0]
        ends=starts[1:]
        starts=starts[:-1]
        vals=x[starts]
        if starts[0]!=0:
            ends=np.concatenate(([starts[0]],ends))
            starts=np.concatenate(([0],starts))
            vals=np.concatenate(([0],vals))
        if ends[-1]!=chr_len[the_chr]:
            starts=np.concatenate((starts,[ends[-1]]))
            ends=np.concatenate((ends,[chr_len[the_chr]]))
            vals=np.concatenate((vals,[0]))
        # write 
        chroms = np.array([the_chr] * len(vals))
        bw_output.addEntries(chroms, starts, ends=ends, values=vals)

    bw_output.close()

mse_all=np.array(mse_all)
pear_all=np.array(pear_all)

score_file.write('avg\t%.5f\t%.5f\n' % (np.mean(mse_all), np.mean(pear_all)))
score_file.close()
print('avg',np.mean(mse_all), np.mean(pear_all))

for the_id in list_dna:
    dict_dna[the_id].close()
for the_id in id_test:
    dict_label[the_id].close()
    dict_feature[the_id].close()













