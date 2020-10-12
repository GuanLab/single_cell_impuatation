import os
import sys
import numpy as np
import pyBigWig

path1='./bigwig/'
path2='./dropdown_bigwig/'
os.system('mkdir -p ' + path2)

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

file_all = os.listdir(path1)
for the_file in file_all:
    print(the_file)
    bw1=pyBigWig.open(path1 + the_file)    
    bw2 = pyBigWig.open(path2 + the_file, 'w')
    bw2.addHeader(list(zip(chr_all , num_bp)), maxZooms=0)
    for the_chr in chr_all:
        print(the_chr)
        x_ori = np.array(bw1.values(the_chr, 0, chr_len[the_chr]))
        starts = np.random.randint(0,chr_len[the_chr] - 114, int(chr_len[the_chr]*1e-3)) 
        tmp = np.zeros(chr_len[the_chr])
        for i in np.arange(114):
            tmp[starts + i] = -1
        x = x_ori + tmp
        x[x<0] = 0
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
        bw2.addEntries(chroms, starts, ends=ends, values=vals)
    bw1.close()
    bw2.close()




