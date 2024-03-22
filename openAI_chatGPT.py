# true: loop was written by the chatgpt 3.5...
# false: it works good

import tensorflow as tf
import numpy as np

# Set the random seeds for reproducibility
seed = 777
np.random.seed(seed)
tf.random.set_seed(seed)

# Generate random 3D TSP data
def generate_tsp_data(num_cities):
    np.random.seed(0)
    return np.random.rand(num_cities, 3)

@tf.function
def tsp_loss(y_true, y_pred):
    y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return tf.reduce_mean(loss)

# Create the model
num_cities = 1000
tsp_data = generate_tsp_data(num_cities)
inputs = tf.keras.layers.Input(shape=(num_cities, 3), batch_size=4)
# lstm1 = tf.keras.layers.GRU(800, return_sequences=True, return_state=True, go_backwards=True)
# lstm2 = tf.keras.layers.SimpleRNN(800, return_sequences=True, return_state=True, go_backwards=True)
# lstm3 = tf.keras.layers.GRU(800, return_sequences=True, return_state=True, go_backwards=True)
# lstm4 = tf.keras.layers.GRU(800, return_sequences=True, return_state=True, go_backwards=True)
# lstm5 = tf.keras.layers.GRU(800, return_sequences=True, return_state=True, go_backwards=True)
# lstm6 = tf.keras.layers.GRU(800, return_sequences=True, return_state=True, go_backwards=True)
lstm1 = tf.keras.layers.Dense(500, activation='elu')
lstm2 = tf.keras.layers.Dense(500, activation='relu')
lstm3 = tf.keras.layers.Dense(500, activation='elu')
lstm4 = tf.keras.layers.Dense(500, activation='relu')
lstm5 = tf.keras.layers.Dense(500, activation='elu')
lstm6 = tf.keras.layers.Dense(500, activation='relu')
last_hidden_state1 = lstm1(inputs)
last_hidden_state2 = lstm2(inputs)
last_hidden_state3= lstm3(inputs)
last_hidden_state4 = lstm4(inputs)
last_hidden_state5 = lstm5(inputs)
last_hidden_state6 = lstm6(inputs)
aaaaa=tf.keras.layers.Concatenate()([last_hidden_state1, last_hidden_state2, last_hidden_state3, last_hidden_state4, last_hidden_state5, last_hidden_state6 ])
middle=tf.keras.layers.Dense(num_cities, activation='relu')(aaaaa)
middle1=tf.keras.layers.Dense(num_cities, activation='elu')(aaaaa)
aaaaa=tf.keras.layers.Concatenate()([middle,middle1])
_,aaaaa= tf.keras.layers.GRU(num_cities, return_sequences=True, return_state=True, go_backwards=True)(aaaaa)
outputs = tf.keras.layers.Dense(num_cities, activation='sigmoid')(aaaaa)
model = tf.keras.Model(inputs, outputs)

# Compile the model
optimizer = tf.keras.optimizers.Adam()
model.summary()
model.compile(optimizer=optimizer, loss=tsp_loss)

# Generate training data
num_samples = 1000
inputs = np.zeros((num_samples, num_cities, 3))
outputs_indices = np.zeros((num_samples, num_cities))
outputs_values = np.zeros((num_samples, num_cities))
batch_size = 4

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

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs_sparse))

# Configure the dataset for optimal performance
dataset = dataset.shuffle(num_samples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Define the training step
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tsp_loss(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    for batch_inputs, batch_targets in dataset:
        loss = train_step(batch_inputs, batch_targets)
        total_loss += loss
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.6f}")
    model.save_weights(f"model_{epoch}_loss_{average_loss:.6f}.weights.h5")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.8f}")
        model.save(f"model_{epoch}_loss_{average_loss}")
