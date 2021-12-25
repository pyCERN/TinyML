import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from tensorflow.keras import layers


samples = 1000
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

x_values = np.random.uniform(low=0, high=2*math.pi, size=samples)
np.random.shuffle(x_values)
y_values = np.sin(x_values) + 0.1*np.random.randn(*x_values.shape)

train_split = int(0.6 * samples)
test_split = int(0.2*samples + train_split)
x_train, x_val, x_test = x_values[:train_split], x_values[train_split:test_split], x_values[test_split:]
y_train, y_val, y_test = y_values[:train_split], y_values[train_split:test_split], y_values[test_split:]

model = tf.keras.models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=600, batch_size=16, validation_data=(x_val, y_val))

# convert w/o quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save tflite model
open('sine_model.tflite', 'wb').write(tflite_model)

# convert with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# specify the dataset for model to make optimization efficiently
def representative_dataset_generator():
    for value in x_test:
        yield [np.array(value, dtype=np.float32, ndmin=2)]

converter.representative_dataset = representative_dataset_generator
tflite_model = converter.convert()

open('sine_model_quantized.tflite', 'wb').write(tflite_model)

"""
Predict with tflite model
1. Interpreter object instantiation
2. Call method allocating memory to the model
3. Input tensor
4. Predict using model
5. Output tensor
"""

# Interpreter object instantiation
sine_model = tf.lite.Interpreter('sine_model.tflite')
sine_model_quantized = tf.lite.Interpreter('sine_model_quantized.tflite')

# Allocating memory
sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()

# Fetch index from input, output tensor
sine_model_input_index = sine_model.get_input_details()[0]['index']
sine_model_output_index = sine_model.get_output_details()[0]['index']
sine_model_quantized_input_index = sine_model_quantized.get_input_details()[0]['index']
sine_model_quantized_output_index = sine_model_quantized.get_output_details()[0]['index']

sine_model_predictions = []
sine_model_quantized_predictions = []

for x_value in x_test:
    x_value_tensor = tf.convert_to_tensor([[x_value]], dtype=np.flaot32)
    sine_model.set_tensor(sine_model_input_index, x_value_tensor)

    # inference
    sine_model.invoke()
    sine_model_predictions.append(sine_model.get_tensor(sine_model_output_index)[0])

    sine_model_quantized.set_tensor(sine_model_quantized_input_index, x_value_tensor)
    sine_model_quantized.invoke()
    sine_model_quantized_predictions.append(sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0])

    import os
    