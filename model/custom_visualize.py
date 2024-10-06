import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data set from CSV file
training_data_df = pd.read_csv("model/sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

# Load testing data set from CSV file
test_data_df = pd.read_csv("model/sales_data_test.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

# All data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# Scale the test data with the same scaler
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Define model parameters
learning_rate = 0.001
training_epochs = 30
number_of_inputs = X_training.shape[1]
number_of_outputs = 1
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Define the model using Keras sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(layer_1_nodes, activation='relu', input_shape=(number_of_inputs,)),
    tf.keras.layers.Dense(layer_2_nodes, activation='relu'),
    tf.keras.layers.Dense(layer_3_nodes, activation='relu'),
    tf.keras.layers.Dense(number_of_outputs)
])

# Compile the model with Adam optimizer and MSE as loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error')

# Create log directories for TensorBoard
log_dir = "./logs/histogram_visualization"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)

# Train the model
history = model.fit(
    X_scaled_training, Y_scaled_training,
    epochs=training_epochs,
    validation_data=(X_scaled_testing, Y_scaled_testing),
    callbacks=[tensorboard_callback]
)

# Evaluate the final model on training and testing data
final_training_cost = model.evaluate(X_scaled_training, Y_scaled_training)
final_testing_cost = model.evaluate(X_scaled_testing, Y_scaled_testing)

print(f"Final Training cost: {final_training_cost}")
print(f"Final Testing cost: {final_testing_cost}")
