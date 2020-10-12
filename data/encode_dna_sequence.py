import pyBigWig
import numpy as np
import os
import sys

def seq_to_hot(seq):
    import numpy as np
    import re
    seq=re.sub('B','N',seq)
    seq=re.sub('[D-F]','N',seq)
    seq=re.sub('[H-S]','N',seq)
    seq=re.sub('[U-Z]','N',seq)
    seq=seq.replace('a','A')
    seq=seq.replace('c','C')
    seq=seq.replace('g','G')
    seq=seq.replace('t','T')
    seq=seq.replace('n','N')
    Aseq=seq
    Aseq=Aseq.replace('A','1')
    Aseq=Aseq.replace('C','0')
    Aseq=Aseq.replace('G','0')
    Aseq=Aseq.replace('T','0')
    Aseq=Aseq.replace('N','0')
    Aseq=np.asarray(list(Aseq),dtype='float32')
    Cseq=seq
    Cseq=Cseq.replace('A','0')
    Cseq=Cseq.replace('C','1')
    Cseq=Cseq.replace('G','0')
    Cseq=Cseq.replace('T','0')
    Cseq=Cseq.replace('N','0')
    Cseq=np.asarray(list(Cseq),dtype='float32')
    Gseq=seq
    Gseq=Gseq.replace('A','0')
    Gseq=Gseq.replace('C','0')
    Gseq=Gseq.replace('G','1')
    Gseq=Gseq.replace('T','0')
    Gseq=Gseq.replace('N','0')
    Gseq=np.asarray(list(Gseq),dtype='float32')
    Tseq=seq
    Tseq=Tseq.replace('A','0')
    Tseq=Tseq.replace('C','0')
    Tseq=Tseq.replace('G','0')
    Tseq=Tseq.replace('T','1')
    Tseq=Tseq.replace('N','0')
    Tseq=np.asarray(list(Tseq),dtype='float32')
    hot=np.vstack((Aseq,Cseq,Gseq,Tseq))
    return hot 

chr_all=[]
for i in range(1,20):
    chr_all.append(str(i))

chr_all.append('X')

num_bp=[195471971,182113224,160039680,156508116,151834684, \
    149736546,145441459,129401213,124595110,130694993, \
    122082543,120129022,120421639,124902244,104043685, \
    98207768,94987271,90702639,61431566,171031299]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

path1='./mg38/'
path2='./dna_npy/'
os.system('mkdir -p ' + path2)

## 1. one-hot
for the_chr in chr_all:
    print(the_chr)
    f=open(path1 + 'Mus_musculus.GRCm38.dna_rm.chromosome.' + the_chr + '.fa')
    line=f.readline()
    seq_ori=''
    for line in f:
        seq_ori += line.rstrip()
    f.close()
    x=seq_to_hot(seq_ori)
    np.save(path2 + 'chr'+ the_chr, x)

path3='./dna_bigwig/'
os.system('mkdir -p ' + path3)

## 2. convert into bigwig
base_all=['A','C','G','T']
for i in np.arange(len(base_all)):
    print(base_all[i])
    bw = pyBigWig.open(path3 + base_all[i] + '.bigwig', 'w')
    bw.addHeader(list(zip(chr_all , num_bp)), maxZooms=0) # zip two turples
    for the_chr in chr_all:
        print(the_chr)
        x=np.load(path2 + 'chr'+ the_chr + '.npy')
        # pad two zeroes
        z=np.concatenate(([0],x[i,:],[0]))
        # find boundary
        tmp1=np.where(np.diff(z)==1)[0]
        tmp2=np.where(np.diff(z)==-1)[0]
        starts=np.concatenate((tmp1, tmp2))
        starts.sort()
        ends=starts[1:]
        starts=starts[:-1]
        vals=np.zeros(len(starts))
        vals[np.arange(0,len(vals),2)]=1 # assume start with 0
        if starts[0]!=0: # if start with 1
            ends=np.concatenate(([starts[0]],ends))
            starts=np.concatenate(([0],starts))
            vals=np.concatenate(([0],vals))
        if ends[-1]!=chr_len[the_chr]: # if end with 0
            starts=np.concatenate((starts,[ends[-1]]))
            ends=np.concatenate((ends,[chr_len[the_chr]]))
            vals=np.concatenate((vals,[0]))
        # write
        chroms = np.array([the_chr] * len(vals))
        bw.addEntries(chroms, starts, ends=ends, values=vals)
    bw.close()



