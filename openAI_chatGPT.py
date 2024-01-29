import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 777
np.random.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Generate random 3D TSP data
def generate_tsp_data(num_cities):
    np.random.seed(0)
    return np.random.rand(num_cities, 3)

@tf.function
def tsp_loss(y_true, y_pred):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, dtype=tf.int32)
    pred_route = tf.one_hot(y_pred, depth=tf.shape(y_true)[-1])
    pred_route = tf.cast(pred_route, dtype=tf.float32)
    y_true = tf.sparse.to_dense(y_true, default_value=0)
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Compute the loss using cross-entropy
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, pred_route)
    return tf.reduce_mean(loss)

# Create the model
num_cities = 1000
tsp_data = generate_tsp_data(num_cities)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(num_cities, input_shape=(num_cities, 3)))
model.add(tf.keras.layers.Dense(num_cities, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Generate training data
num_samples = 1000
inputs = np.zeros((num_samples, num_cities, 3))
outputs_indices = np.zeros((num_samples, num_cities))
outputs_values = np.zeros((num_samples, num_cities))
with tf.device('/device:GPU:0'):
    for i in range(num_samples):
        permutation = np.random.permutation(num_cities)
        inputs[i] = tsp_data[permutation]
        outputs_indices[i] = np.arange(num_cities)
        outputs_values[i] = permutation

    outputs_indices = outputs_indices.astype(np.int64)  # Convert to int64 data type
    outputs_values = outputs_values.astype(np.int64)  # Convert to int64 data type

    # Convert to sparse representation
    indices = np.stack([np.repeat(np.arange(num_samples), num_cities), outputs_indices.flatten()]).T
    values = outputs_values.flatten()
    dense_shape = (num_samples, num_cities)
    outputs_sparse = tf.sparse.SparseTensor(indices, values, dense_shape)
    batch_size = 4
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs_sparse))

    # Configure the dataset for optimal performance
    dataset = dataset.shuffle(num_samples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Define the training step
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = tsp_loss(predictions,outputs_sparse)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Define the training loop
    num_epochs = 10000

    optimizer = tf.keras.optimizers.Adam()
    my_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=r'./chatNN_epoka-{epoch:02d}_loss-{loss:.6f}.h5', verbose=1, monitor="loss", save_weights_only=False, mode="min", save_best_only=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_inputs, batch_targets in dataset:
            loss = train_step(batch_inputs, batch_targets)
            total_loss += loss
            num_batches += 1
        average_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.6f}")

        # Add any additional callbacks or save checkpoints here
