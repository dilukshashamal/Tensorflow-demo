import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

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

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Define the model using Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_dim=9, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)  # output layer
])

# Compile the model with Adam optimizer and MSE loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error')

# Create TensorBoard callback to log training and testing data
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom callback to log test data
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, log_dir):
        super(TestCallback, self).__init__()
        self.test_data = test_data
        self.test_writer = tf.summary.create_file_writer(os.path.join(log_dir, "test"))

    def on_epoch_end(self, epoch, logs=None):
        test_loss = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        with self.test_writer.as_default():
            tf.summary.scalar('loss', test_loss, step=epoch)

# Train the model and log training/testing data
test_callback = TestCallback(test_data=(X_scaled_testing, Y_scaled_testing), log_dir=log_dir)
history = model.fit(X_scaled_training, Y_scaled_training, 
                    epochs=100, 
                    callbacks=[tensorboard_callback, test_callback],
                    verbose=2)

# Final evaluation of the model
final_training_loss = model.evaluate(X_scaled_training, Y_scaled_training, verbose=0)
final_testing_loss = model.evaluate(X_scaled_testing, Y_scaled_testing, verbose=0)

print(f"Final Training loss: {final_training_loss}")
print(f"Final Testing loss: {final_testing_loss}")

# Make predictions using the test data
Y_predicted_scaled = model.predict(X_scaled_testing)
Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

# Output prediction results for the first sample
real_earnings = test_data_df['total_earnings'].values[0]
predicted_earnings = Y_predicted[0][0]

print(f"The actual earnings of Game #1 were ${real_earnings}")
print(f"Our neural network predicted earnings of ${predicted_earnings}")

