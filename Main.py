import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp 
print("Reading CSV file...")
df = pd.read_csv("C:/Users/Shaurya/Desktop/US_Accidents_March23.csv")
#print(list(df))
#dataplot=sb.heatmap(df.corr())
dfresult = df.dropna()
print(list(dfresult))
print(list(df))