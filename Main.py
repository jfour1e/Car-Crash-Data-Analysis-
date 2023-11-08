#Main 

#imports 
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

df = pd.read_csv('US_Accidents_March23.csv')
#df = pd.read_csv("C:/Users/Shaurya/Desktop/US_Accidents_March23.csv")

#print(list(df))
#dataplot=sb.heatmap(df.corr())
dfresult = df.dropna()
print(list(dfresult))
print(list(df))