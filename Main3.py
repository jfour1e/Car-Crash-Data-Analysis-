from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

df = pd.read_csv('US_Accidents_March23.csv')
df = df[['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Street', 'City', 'County', 'State', 'Zipcode', 'Temperature(F)','Wind_Chill(F)', 'Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Crossing', 'Junction', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset']]

#clean data 
df['Temperature(F)'].fillna(df['Temperature(F)'].mean(), inplace=True)
df['Visibility(mi)'].fillna(df['Visibility(mi)'].mean(), inplace=True)
df['Stop'] = df['Stop'].astype(int)

#one-hot encode the weather condition column 
df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)

# Select relevant columns
features = ['Temperature(F)', 'Stop', 'Visibility(mi)'] + list(df.filter(regex='WeatherCondition').columns)
target = 'Severity'

# Create feature matrix (X) and target vector (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

"""
#Initialize a second random forest regressor 
rf_model2 = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the model
rf_model2.fit(X_train, y_train)

# Make predictions on the test set
predictions_model2 = rf_model2.predict(X_test)

# Evaluate the model
mse2 = mean_squared_error(y_test, predictions_model2)
print(f'Mean Squared Error: {mse2}')
"""

