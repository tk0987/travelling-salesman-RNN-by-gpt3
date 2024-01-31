# i've made this folder private, because i've mistaken a source file...

import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
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

# Create the model - it is a scrap, but working scrap - 18x better than connecting points 'row_wise'
# kinda crude, and scrap - beware
num_cities = 1000
tsp_data = generate_tsp_data(num_cities)
inputs = tf.keras.layers.Input(shape=(num_cities, 3), batch_size=4)
lstm1 = tf.keras.layers.SimpleRNN(100, return_sequences=False, return_state=True, input_shape=(num_cities, 3))
lstm2 = tf.keras.layers.SimpleRNN(100, return_sequences=False, return_state=True, input_shape=(num_cities, 3))
_1, last_hidden_state1= lstm1(inputs)
_2, last_hidden_state2= lstm2(inputs)
# sum1=tf.keras.layers.Add()([last_hidden_state1,last_hidden_state2])
lstm3 = tf.keras.layers.GRU(100, return_sequences=False, return_state=True, input_shape=(num_cities, 3))
lstm4 = tf.keras.layers.SimpleRNN(100, return_sequences=False, return_state=True, input_shape=(num_cities, 3))
_3, last_hidden_state3= lstm3(inputs)
_4, last_hidden_state4= lstm4(inputs)
lstm5 = tf.keras.layers.LSTM(100, return_sequences=False, return_state=True, input_shape=(num_cities, 3))
_51,asdfgt,_=lstm5(inputs)
sum2=tf.keras.layers.Add()([_3,_4,_1,_2,_51])
hid_lay=tf.keras.layers.Dense(300, activation="relu")
_5= hid_lay(sum2)
outputs = tf.keras.layers.Dense(num_cities, activation='softmax')(_5)
model = tf.keras.Model(inputs, outputs)
model.summary()
# Compile the model
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=tsp_loss)

# Generate training data
num_samples = 10
inputs = [generate_tsp_data(num_cities) for i in range (num_samples)] # i found more relevant version of this code, and this corrected line seems to make more sense! what a mess
outputs_indices = np.zeros((num_samples, num_cities))
outputs_values = np.zeros((num_samples, num_cities))
batch_size = 4
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
    num_epochs = 10000
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_inputs, batch_targets in dataset:
            loss = train_step(batch_inputs, batch_targets)
            total_loss += loss
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.6f}")
        tf.saved_model.save(model,"./my_recurrent")
