import argparse
import numpy as np
import pyBigWig

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()


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
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument('-f', '--fold', default='0', type=int, help='fold for test partition')
    args = parser.parse_args()
    return args

args=get_args()

print(args)
fold_partition=args.fold

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

score_file=open('baseline_fold' + str(fold_partition) + '.txt','w')
score_file.write('id\tmse\tpear\n')

mse_all=[]
pear_all=[]

for id_target in id_test:
    #list_chr = chr_all
    list_chr = ['19'] #TODO
    for the_chr in list_chr:
        print('chr' + the_chr)
        pred_final=np.array(dict_feature[id_target].values(the_chr,0,chr_len[the_chr]))
        gt=np.array(dict_label[id_target].values(the_chr,0,chr_len[the_chr]))
        ## scoreing
        the_mse = mse(gt, pred_final)
        the_pear = np.corrcoef(gt, pred_final)[0,1]
        score_file.write('%s\t%.5f\t%.5f\n' % (id_target, the_mse, the_pear))
        score_file.flush()
        print(id_target, the_mse, the_pear)
        mse_all.append(the_mse)
        pear_all.append(the_pear)

mse_all=np.array(mse_all)
pear_all=np.array(pear_all)

score_file.write('avg\t%.5f\t%.5f\n' % (np.mean(mse_all), np.mean(pear_all)))
score_file.close()
print('avg',np.mean(mse_all), np.mean(pear_all))

for the_id in id_test:
    dict_label[the_id].close()
    dict_feature[the_id].close()



#the_id = 'GTCTTCGCAAATCCGT-1'
#x=np.load('pred_chr19_GTCTTCGCAAATCCGT-1.npy')
#np.max(x)
##162.406982421875
#np.min(x)
##0.0
#np.mean(x)
##0.002100773913296661
#bw=pyBigWig.open('../../data/bigwig/GTCTTCGCAAATCCGT-1.bigwig')
#gt = np.array(bw.values('19',0,61431566))
#bw1=pyBigWig.open('../../data/dropdown_bigwig/GTCTTCGCAAATCCGT-1.bigwig')
#x1 = np.array(bw1.values('19',0,61431566)
#
#
#np.max(x1)
#151.0
#np.max(gt)
#151.0
#
#
#mse(x,gt)
##0.002052628508694983
#mse(x1,gt)
##6.677348905609862e-05
#
#np.corrcoef(x,gt)[0,1]
##0.9912569095036418
#np.corrcoef(x1,gt)[0,1]
##0.9996871012644398
#
#zero = np.zeros(61431566)
#avg = zero + np.mean(gt)
#mse(zero, gt)
##0.10661375619172724
#mse(avg, gt)
##0.10660646753024283
#
#
#
#the_id = 'GTAACTGGTTCCCGAG-1'
#x=np.load('pred_chr19_' + the_id + '.npy')
#np.max(x);np.min(x);np.mean(x)
##57.13926696777344
##0.0
##0.0020559306747051686
#bw=pyBigWig.open('../../data/bigwig/' + the_id + '.bigwig')
#gt = np.array(bw.values('19',0,61431566))
#bw1=pyBigWig.open('../../data/dropdown_bigwig/' + the_id + '.bigwig')
#x1 = np.array(bw1.values('19',0,61431566))
#
#np.max(x1);np.max(gt)
#np.mean(x1);np.mean(gt)
#
#mse(x,gt)
##0.0015104053592938317
#mse(x1,gt)
##6.985008326175504e-05
#
#np.corrcoef(x,gt)[0,1]
##0.9816510842340911
#np.corrcoef(x1,gt)[0,1]
##0.9990440272097221
#
#zero = np.zeros(61431566)
#avg = zero + np.mean(gt)
#mse(zero, gt)
##0.03572349107948835
#mse(avg, gt)
##0.035716078846183795
#
#
#
#
#
