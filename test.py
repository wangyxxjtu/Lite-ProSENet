import numpy as np
label = 9
Thresh = 3
Survival_time = int(100/365.0 + 0.5)
if Survival_time >= Thresh:
    print(label)
else:
    print(Thresh)