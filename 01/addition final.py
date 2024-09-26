import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

X = tf.constant([2, 4, 6], name="X")
Y = tf.constant([4, 1, 3], name="Y")

addition = tf.add(X, Y, name="addition")

print(addition.numpy())