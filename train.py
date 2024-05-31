import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset_dir = "dataset.csv"
df = pd.read_csv(dataset_dir)

df['Country'].nunique()
df_ph = df[df['Country'] == 'Philippines']
df_ph['AvgTemperature'] = pd.to_numeric(
    df_ph['AvgTemperature'], errors='coerce')
df_ph.dropna(subset=['AvgTemperature'], inplace=True)
sns.boxplot(x=df_ph['AvgTemperature'])
plt.title(f'Boxplot for AvgTemperature')
plt.show()
Q1 = df_ph['AvgTemperature'].quantile(0.25)
Q3 = df_ph['AvgTemperature'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering outliers
no_outliers_df = df_ph[(df_ph['AvgTemperature'] >= lower_bound) & (
    df_ph['AvgTemperature'] <= upper_bound)]

# plot the boxplot again to confirm that outliers are removed
sns.boxplot(x=no_outliers_df['AvgTemperature'])
plt.title(f'Boxplot without outliers for AvgTemperature')
plt.show()

combined_date_df = no_outliers_df.copy()
combined_date_df['Date'] = pd.to_datetime(
    no_outliers_df[['Year', 'Month', 'Day', 'Hour']])
combined_date_df.head()
# Drop unnecessary columns
preprocessed_df = combined_date_df.drop(
    ['Country', 'Month', 'Day', 'Year', 'Hour'], axis=1)
# Reorder the columns
preprocessed_df = preprocessed_df[[
    'Date', 'AvgTemperature', 'Precipitation', 'Status']]
print("Shape: ", preprocessed_df.shape)
preprocessed_df.head()
preprocessed_df.to_csv('preprocessed_data.csv', index=False)

# Extracting the 'AvgTemperature' column as the feature 'X'
X = preprocessed_df['AvgTemperature'].values.reshape(-1, 1)

# Getting the length of the dataset
dataset_length = len(preprocessed_df)

# Calculating the number of data points for each split
train_size = int(0.7 * dataset_length)
val_size = int(0.15 * dataset_length)
test_size = int(0.15 * dataset_length)

# Creating training, validation, and test datasets
training_data = X[:train_size]
val_data = X[train_size:train_size + val_size]
test_data = X[train_size + val_size:]

# Print information about the splits
print(f"Total dataset length: {dataset_length}")
print(f"Training set length: {train_size}")
print(f"Validation set length: {val_size}")
print(f"Test set length: {test_size}")

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_data)
val_set_scaled = sc.fit_transform(val_data)
test_set_scaled = sc.fit_transform(test_data)


input_size = 100

# Test data -------------------------------------------------------------
X_train = []
y_train = []
for i in range(input_size, len(training_data)):
    X_train.append(training_set_scaled[i-input_size:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Validation Data -------------------------------------------------------
X_val = []
y_val = []
for i in range(input_size, len(val_data)):
    X_val.append(val_set_scaled[i-input_size:i, 0])
    y_val.append(val_set_scaled[i, 0])

X_val, y_val = np.array(X_val), np.array(y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Test Data -------------------------------------------------------------
X_test = []
y_test = []
for i in range(input_size, len(test_data)):
    X_test.append(test_set_scaled[i-input_size:i, 0])
    y_test.append(test_set_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(Dropout(.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(units=50))
model.add(Dropout(.2))
model.add(Dense(units=1))
model.summary()

callbacks = [
    EarlyStopping(patience=50, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.000001, verbose=1),
]

model.compile(optimizer='adam', loss= 'mean_absolute_error')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=callbacks,
    batch_size=32
)

model.save("final_custom_weather_data.keras")
