# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:32:19 2019

@author: mücahit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

veriler=pd.read_csv("Ads_Ctr_Optimisation.csv")
"""
import random 

N=10000
d=10
toplam=0
secilenler=[]

for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    toplam=toplam + odul 
    
plt.hist(secilenler)
plt.show()"""

#UCB

N=10000 # TİKLAMA
d=10 #toplam 10 ilan var

#Ri(n)
rewards=[0] * d #ilk başta bütün ilanların odülü 0
#Ni(n)
clicks=[0] * d # o ana kadar ki tiklamalar
summar=0 #toplam odul 
selections=[]
for n in range(1,N):
    ad=0 #secilen ilan
    max_ucb=0
    for i in range (0,d):
        if(clicks[i]>0):
            average=rewards[i] / clicks[i]
            delta=math.sqrt(3/2 * math.log(n)/ clicks[i])
            ucb=average + delta
        else:
            ucb=N*10
        if max_ucb <ucb:
            max_ucb=ucb
            ad=i    
    selections.append(ad)
    clicks[ad]=clicks[ad]+1
    odul=veriler.values[n,ad]
    rewards[ad]=rewards[ad] + odul 
    summar=summar +  odul 
print("Toplam odul")
print(summar)

plt.hist(selections)
plt.show()
    
        