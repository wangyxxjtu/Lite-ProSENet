import numpy as np
label = np.array([1,2,4,5,3])
label = np.array([1,2,3,4,5])
event = np.array([1,1,0,0,1])
prediction= np.array([1,2,4,3,5])
prediction= np.array([1,2,3,4,5])


# from sksurv.metrics import (concordance_index_censored,
#                             concordance_index_ipcw,
#                             cumulative_dynamic_auc)

import itertools
com_list=list(itertools.combinations([0,1,2,3,4],2))
print(com_list)


count = 0
total = 0
for (i,j) in com_list:
    # print(i,j)
    if j != i:
        if event[i] * event[j]==1:
            print(label[i],label[j],prediction[j])
            if label[i]<=label[j] and label[i]<=prediction[j]:
                count +=1
                total +=1
            elif label[i]>label[j] and label[i]>prediction[j]:
                count +=1
                total +=1
            else:
                total +=1
        if event[j]+event[i]==1:
            print(label[i],label[j],prediction[j])
            if event[j]==0 and label[i]<=label[j]:
                    if label[i]<=prediction[j]:
                        count +=1
                        total +=1
                    else:
                        total +=1
            if event[i]==0 and label[i]>label[j]:
                    if label[i]>prediction[j]:
                        count +=1
                        total +=1
                    else:
                        total +=1
print(count,total,count/total)

# def concordance(label,prediction):
    # 
    # return score