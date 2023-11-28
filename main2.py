import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


df = pd.read_csv('US_Accidents_March23.csv')

df = df.drop(['End_Lat','End_Lng','ID','Source'],axis='columns')

df = df.fillna(df.mean())

# Example: Removing punctuation
df['Description'] = pd.to_numeric(df['Description'], errors='coerce')
df['Street'] = pd.to_numeric(df['Street'], errors='coerce')
df['City'] = pd.to_numeric(df['City'], errors='coerce')
df['County'] = pd.to_numeric(df['County'], errors='coerce')
df['State'] = pd.to_numeric(df['State'], errors='coerce')
df['Timezone'] = pd.to_numeric(df['Timezone'], errors='coerce')
df['Airport_Code'] = pd.to_numeric(df['Airport_Code'], errors='coerce')
df['Sunrise_Sunset'] = pd.to_numeric(df['Sunrise_Sunset'], errors='coerce')
df['Civil_Twilight'] = pd.to_numeric(df['Civil_Twilight'], errors='coerce')
df['Nautical_Twilight'] = pd.to_numeric(df['Nautical_Twilight'], errors='coerce')
df['Astronomical_Twilight'] = pd.to_numeric(df['Astronomical_Twilight'], errors='coerce')
df['Weather_Condition'] = pd.to_numeric(df['Weather_Condition'], errors='coerce')
df['Wind_Direction'] = pd.to_numeric(df['Wind_Direction'], errors='coerce')
df['Zipcode'] = pd.to_numeric(df['Zipcode'], errors='coerce')
df['Country'] = pd.to_numeric(df['Country'], errors='coerce')


# Convert 'Weather_Timestamp' to datetime with error handling
df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')

# Convert 'Start_Time' to datetime with error handling
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

# Convert 'End_Time' to datetime with error handling
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Choose a reference date (e.g., '1970-01-01 00:00:00')
reference_date = pd.to_datetime('1970-01-01 00:00:00')

# Calculate the numeric values (number of seconds since the reference date)
df['Weather_Timestamp_Numeric'] = (df['Weather_Timestamp'] - reference_date).dt.total_seconds()
df['Start_Time_Numeric'] = (df['Start_Time'] - reference_date).dt.total_seconds()
df['End_Time_Numeric'] = (df['End_Time'] - reference_date).dt.total_seconds()

correlation_matrix = df.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(23, 35))
sb.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
