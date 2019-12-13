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
import random 
"""


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


summar=0 #toplam odul 
selections=[]
birler=[0]*d
sifirlar=[0]*d
for n in range(1,N):
    ad=0 #secilen ilan
    max_th=0
    for i in range (0,d):
        rasbeta=random.betavariate(birler[i]+1 , sifirlar[i]+1)
        if(rasbeta>max_th):
            max_th=rasbeta
            ad=i
    selections.append(ad)
    odul=veriler.values[n,ad]
    if odul ==1:
        birler[ad]=birler[ad]+1
    else :
        sifirlar[ad]=sifirlar[ad]+1
    summar=summar+odul
print("Toplam odul")
print(summar)

plt.hist(selections)
plt.show()
    
        