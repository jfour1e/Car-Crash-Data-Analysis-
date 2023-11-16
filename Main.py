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

#dataplot=sb.heatmap(df.corr())
#dfresult = df.dropna()
#print(list(dfresult))
#print(list(df))

df1 = df.drop(['Source', 'Description', 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Direction', 'Wind_Speed(mph)', 'Bump', 'Amenity', 'No_Exit', 'Give_Way', 'Railway', 'Roundabout', 'Station','Traffic_Calming', 'Turning_Loop', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'],axis='columns')
print(df1.columns)

