import numpy as np

id_all=np.loadtxt('id_all.txt', dtype='str')

id_all.sort()
print(id_all)
np.random.seed(449) # 5-fold
np.random.shuffle(id_all)
num = int(np.ceil(len(id_all)*0.2))
for i in np.arange(5):
    start = i * num
    end = (i+1) * num
    if end > len(id_all):
        end = len(id_all)
    id_test = id_all[start:end]
    id_test.sort()
    np.savetxt('id_test' + str(i) + '.txt', id_test, fmt='%s')



