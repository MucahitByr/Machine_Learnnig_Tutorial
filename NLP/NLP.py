# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:44:47 2019

@author: mücahit
"""
import pandas as pd
import numpy as np

veriler=pd.read_csv("Restaurant_Reviews.csv")
import re

veri=re.sub("[^a-zA-Z]", " " , veriler["Review"][6])
print(veri)

yorum=veri.lower()
yourm=veri.split()