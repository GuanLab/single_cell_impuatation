#!/bin/bash

set -e


time python train.py -f 0
# real	6m8.043s


time python predict.py -m weights_fold0_seed0.h5 -f 0


## simply score the input as the baseline
time python score_baseline.py -f 0 
#real	8m50.350s


#sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g; s/the_lr=1e-3/the_lr=1e-4/g; s/model.summary()/#model.summary()/g' train.py > continue_train.py



