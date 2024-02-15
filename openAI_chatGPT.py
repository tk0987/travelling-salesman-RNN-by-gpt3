# original training loop restored. loss function will need to be updated. architecture 2x better than connecting points row/col-wise after first epoch
# upgrading cuda for tf 2.15 - after this model will be tested after more epochs
import tensorflow as tf
import numpy as np
import os
# import h5py
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
with tf.device('/device:GPU:0'):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Set the random seeds for reproducibility
    seed = 777
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Generate random 3D TSP data
    def generate_tsp_data(num_cities):
        np.random.seed(seed)
        return np.random.rand(num_cities, 3)

    @tf.function
    def tsp_loss(y_true, y_pred):
        y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
        loss = tf.keras.losses.huber(y_true,y_pred[:])
        return tf.reduce_mean(loss)
    global num_cities

    # Create the model
    num_cities = 512
    tsp_data = generate_tsp_data(num_cities)
    def model1(n,num_cities):
        inputs = tf.keras.layers.Input(shape=(num_cities, 3), batch_size=1)
        # inputs=tf.keras.layers.Input(shape=(num_cities, 3), batch_size=1)
        nd1,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd2,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd3,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd4,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd5,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd6,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd7,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

        rd1,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd2,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd3,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd4,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd5,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd6,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd7,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

        sum1=tf.keras.layers.Add()([rd1,rd2,rd3,rd4,rd5,rd6,rd7])
        sum2=tf.keras.layers.Add()([nd1,nd2,nd3,nd4,nd5,nd6,nd7])

        middle1=tf.keras.layers.Dense(num_cities,"elu")(sum1)
        middle2=tf.keras.layers.Dense(num_cities,"relu")(sum2)

        middle=tf.keras.layers.Add()([middle1,middle2])
        # middle=tf.keras.layers.Flatten()(middle)
        outputs=tf.keras.layers.Dense(1,"softmax")(middle)
        # print(tf.shape)
        return tf.keras.Model(inputs, outputs)
    model=model1(44,num_cities=num_cities)
    model.summary()
    # Compile the model
    optimizer = tf.keras.optimizers.AdamW(0.0005, 0.007, 0.8, 0.98, 1e-6)
    model.compile(optimizer=optimizer, loss=tsp_loss)

    # Generate training data
    num_samples = 1000
    inputs = np.zeros((num_samples, num_cities, 3))
    outputs_indices = np.zeros((num_samples, num_cities))
    outputs_values = np.zeros((num_samples, num_cities))
    batch_size = 1

    for i in range(num_samples):
        permutation = np.random.permutation(num_cities)
        inputs[i] = tsp_data[permutation]
        outputs_indices[i] = np.arange(num_cities)
        outputs_values[i] = permutation

    outputs_indices = outputs_indices.astype(np.int32)  # Convert to int64 data type
    outputs_values = outputs_values.astype(np.float32)  # Convert to int64 data type

    # Convert to sparse representation
    indices = np.stack([np.repeat(np.arange(num_samples), num_cities), outputs_indices.flatten()]).T
    values = outputs_values.flatten()
    dense_shape = (num_samples, num_cities)
    outputs_sparse = tf.sparse.SparseTensor(indices, values, dense_shape)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs_sparse))

    # Configure the dataset for optimal performance
    dataset = dataset.shuffle(num_samples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # print(tf.shape(dataset))
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
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_inputs, batch_targets in dataset:
            loss = train_step(batch_inputs, batch_targets)
            total_loss += loss
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.8f}")
        model.save(f"model_{epoch}_loss_{average_loss}")
