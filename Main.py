#Main 

#imports 
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

#dataplot=sb.heatmap(df.corr())
#dfresult = df.dropna()
#print(list(dfresult))
#print(list(df))

#df1 = df.drop(['Source', 'Description', 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Direction', 'Wind_Speed(mph)', 'Bump', 'Amenity', 'No_Exit', 'Give_Way', 'Railway', 'Roundabout', 'Station','Traffic_Calming', 'Turning_Loop', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'],axis='columns')
#print(df1.columns)

df = pd.read_csv('US_Accidents_March23.csv')
df = df[['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Street', 'City', 'County', 'State', 'Zipcode', 'Temperature(F)','Wind_Chill(F)', 'Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Crossing', 'Junction', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset']]

print(df['Start_Time'].dtype)

df = df[['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Street', 'City', 'County', 'State', 'Zipcode', 'Temperature(F)','Wind_Chill(F)', 'Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Crossing', 'Junction', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset']]
#df = pd.read_csv("C:/Users/Shaurya/Desktop/US_Accidents_March23.csv")


#split the zipode columns by mailing regions
df[['Zipcode', 'Mailing_Zip']] = df['Zipcode'].str.split('-', expand=True)

#new dataframe to find density of car crashes per county 
county_entry_counts = df.groupby('Zipcode').size().reset_index(name='entry_count')

county_entry_counts['entry_count_scaled'] = np.log10(county_entry_counts['entry_count'])


#import the geojson for US counties 
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


#create the plotly express density map 

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'firefox'

fig = px.choropleth_mapbox(
    county_entry_counts,
    geojson=counties,
    locations= county_entry_counts['Zipcode'],  # Replace 'FIPS' with the actual column name in your DataFrame
    color='entry_count_scaled',  # Replace 'Density' with the column you want to visualize
    hover_name='Zipcode',  # Replace 'County' with the column containing county names
    mapbox_style="carto-positron",
    center=dict(lat=37.0902, lon=-95.7129),
    zoom=3,
    opacity=0.7,
    title='Density Car Crash Map of the Continental US'
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title='Density'
        # Adjust tick labels as needed
    )
)

fig.show()

#dataplot=sb.heatmap(df.corr())
#dfresult = df.dropna()
#print(list(dfresult))
#print(list(df))

