import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data set from CSV file
training_data_df = pd.read_csv("02/sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

# Load testing data set from CSV file
test_data_df = pd.read_csv("02/sales_data_test.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

# All data needs to be scaled to a small range like 0 to 1 for the neural network to work well.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# Scale the testing data with the same scaler
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Define model parameters
learning_rate = 0.001
training_epochs = 30

# Define model architecture using Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_dim=9, activation='relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(50, activation='relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_uniform')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

# Set up the TensorBoard callback
log_dir = "02/logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with TensorBoard callback
history = model.fit(X_scaled_training, Y_scaled_training, 
                    epochs=training_epochs, 
                    validation_data=(X_scaled_testing, Y_scaled_testing),
                    callbacks=[tensorboard_callback])

# Save the model
model.save('02/logs/trained_model.h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('02/logs/trained_model.h5')
print("Trained model loaded from disk.")

# Evaluate the model
training_loss = loaded_model.evaluate(X_scaled_training, Y_scaled_training)
testing_loss = loaded_model.evaluate(X_scaled_testing, Y_scaled_testing)

print(f"Final Training loss: {training_loss}")
print(f"Final Testing loss: {testing_loss}")

# Make predictions
Y_predicted_scaled = loaded_model.predict(X_scaled_testing)

# Unscale the predictions
Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

# Display the first prediction
real_earnings = test_data_df['total_earnings'].values[0]
predicted_earnings = Y_predicted[0][0]

print(f"The actual earnings of Game #1 were ${real_earnings}")
print(f"Our neural network predicted earnings of ${predicted_earnings}")
