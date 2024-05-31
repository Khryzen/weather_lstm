import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the historical data
historical_data = pd.read_csv(
    'preprocessed_data.csv', parse_dates=['Date'], index_col='Date')

# Ensure the 'AvgTemperature' column exists in the DataFrame
if 'AvgTemperature' not in historical_data.columns:
    raise ValueError(
        "The 'AvgTemperature' column is not present in the historical data CSV file.")

# Extract the AvgTemperature values
values = historical_data[['AvgTemperature']].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# Define the number of steps to predict
steps_to_predict = 365  # Assuming you want to predict the entire year of 2024

# Get the last input data
input_data = scaled_values[-100:]  # Assuming the input_size is 100

# Reshape input data to match the model input shape
input_data = np.reshape(input_data, (1, input_data.shape[0], 1))

# Load the trained model
loaded_model = tf.keras.models.load_model(
    'custom_weather_data.keras')

predicted_values = []

# Iterate to predict the next steps
for _ in range(steps_to_predict):
    prediction = loaded_model.predict(input_data)
    predicted_values.append(prediction[0][0])
    input_data = np.append(input_data[:, 1:, :], [[prediction[0]]], axis=1)

predicted_values = scaler.inverse_transform(
    np.array(predicted_values).reshape(-1, 1))
date_range_2024 = pd.date_range(
    start='2024-01-01', periods=steps_to_predict)
predicted_df = pd.DataFrame(predicted_values, index=date_range_2024, columns=[
                            'Predicted AvgTemperature'])
predicted_df.to_csv(
    'predictions_2024.csv', index_label='Date')
print("Predictions for the entire year of 2024 have been saved as 'predictions_2024.csv'")
