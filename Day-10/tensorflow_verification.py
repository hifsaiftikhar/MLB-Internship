import tensorflow as tf
from tensorflow import keras

# 1. Verify the installation
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# 2. Confirm TensorFlow and Keras imported successfully
print("\nTensorFlow and Keras imported successfully.")

# 3. Quick check: TensorFlow can create and run a basic tensor operation
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
result = tf.add(a, b)
print("\nSample tensor operation (a + b):", result.numpy())

# 4. Check if a GPU is available (not required, just informational)
gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
print("\nGPU available:", gpu_available)
if not gpu_available:
    print("Running on CPU - this is fine for small models like the ones in today's practice.")
