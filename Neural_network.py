import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2

# reading the database and storing it in a varibale
df = pd.read_csv('US_Accidents_March23.csv')

df = df.drop(['End_Lat','End_Lng','ID','Source'],axis='columns')
print(df.columns)

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

# Selecting the relevant columns
#   'Severity' is the target variable
y = df['Severity']
X = df[['Visibility(mi)', 'Temperature(F)', 'Weather_Condition', 'Stop']]
X['Stop'] = X['Stop'].astype(int)

# Standardizing numeric features
numeric_columns = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X.loc[:, numeric_columns] = scaler.fit_transform(X[numeric_columns])

# One-hot encoding for categorical features
encoder = OneHotEncoder()
weather_condition_encoded = encoder.fit_transform(X[['Weather_Condition']]).toarray()

# Drop the original 'Weather_Condition' column
X = X.drop('Weather_Condition', axis=1)

# Concatenate the encoded data (the weather condition)
X = np.concatenate([X, weather_condition_encoded], axis=1)

# One-hot encode the target variable
severity_encoder = OneHotEncoder(sparse=False)
y_encoded = severity_encoder.fit_transform(y.values.reshape(-1, 1))

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to float32 so everything is the same data type
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')


# Building the neural network model
model = keras.Sequential([
    # Input layer
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(X_train.shape[1],)),
    BatchNormalization(),

    # Additional hidden layers
    Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    BatchNormalization(),
    Dropout(0.3),
    # Output layer
    Dense(y_encoded.shape[1], activation='softmax')
])


keras.layers.BatchNormalization(),

# Compiling the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_accuracy')
lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)


# Training the model
history = model.fit(
    X_train, y_train, 
    epochs=100,  # a large number of epochs and let early stopping decide when to stop
    validation_split=0.2, 
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint, lr_schedule]
)

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plotting training and validation loss/accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Loss/Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()