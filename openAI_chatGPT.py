import tensorflow as tf
import numpy as np
import os
from datetime import datetime
# import h5py
with tf.device('/device:CPU:0'):
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeEr
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Set the random seeds for reproducibility
    seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # def generate_tsp_data(num_cities):
    #     np.random.seed(seed)
    #     return np.random.rand(num_cities, 3)
    global inputs
    global num_cities
    num_cities = 4096
    global num_samples
    num_samples = 1000
   
    def model1(n,num_cities):
        inputs = tf.keras.layers.Input(shape=(num_cities, 3), batch_size=1)
        # inputs=tf.keras.layers.Input(shape=(num_cities, 3), batch_size=1)
        nd1,asdf=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd2,asdf=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd3,asdf=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd4,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd5,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd6,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd7,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

        # rd1=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        # rd2=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        # rd3=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        # rd4=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        # rd5=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        # rd6=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        # rd7=tf.keras.layers.Dense(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

        # sum1=tf.keras.layers.Add()([rd1,rd2,rd3,rd4,rd5,rd6,rd7])
        sum2=tf.keras.layers.Add()([nd1,nd2,nd3,nd4,nd5,nd6,nd7])
        sum2=tf.keras.layers.Dense(n,"elu")(sum2)

        rd1,asdf,sdfghsf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd2,asdf,sdfgsdfg=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd3,asdf=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd4,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd5,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd6,asdf=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd7=tf.keras.layers.Dense(n,"elu")(inputs)

        sum3=tf.keras.layers.Add()([rd1,rd2,rd3,rd4,rd5,rd6,rd7])

        middle1=tf.keras.layers.Dense(num_cities,"relu")(sum3)
        middle2=tf.keras.layers.Dense(num_cities,"relu")(sum2)

        middle=tf.keras.layers.Add()([middle1,middle2])
        middle=tf.keras.layers.Dense(num_cities,"relu")(middle)
        # middle=tf.keras.layers.Flatten()(middle)
        outputs=tf.keras.layers.Dense(1,"sigmoid")(middle)
        # print(tf.shape)
        return tf.keras.Model(inputs, outputs)



    @tf.function
    def tsp_loss(y_true, y_pred):
        # y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
        loss = tf.keras.losses.mean_squared_error(y_true,y_pred[:])
        return tf.reduce_mean(loss)
    @tf.function
    def travelling_loss(y_true,y_pred):
        indices=tf.argsort(y_pred)
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true,inputs[indices]))
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = tsp_loss(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    # Generate random 3D TSP data

    # global num_cities

    # Create the model

    
    model=model1(5,num_cities=num_cities)
    model.summary()
    # Compile the model
    optimizer = tf.keras.optimizers.AdamW(0.01, 0.8)
    model.compile(optimizer=optimizer, loss=travelling_loss)

    # Generate training data
    
    

    # Define the training step


# Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # num_cities = np.random.randint(3,65535)
        def generate_tsp_data(num_cities):
            seed = int(datetime.now().timestamp())
            np.random.seed(seed)
            return np.random.rand(num_cities, 3)
        tsp_data = generate_tsp_data(num_cities)
        inputs = np.asanyarray([generate_tsp_data(num_cities=num_cities) for i in range(num_samples)])
        outputs_indices = np.zeros((num_samples, num_cities))
        outputs_values = np.zeros((num_samples, num_cities))
        batch_size = 1

        for i in range(num_samples):
            permutation = np.random.permutation(num_cities)
            # inputs[i] = tsp_data[permutation]
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
        total_loss = 0.0
        num_batches = 0
        for batch_inputs, batch_targets in dataset:
            loss = train_step(tf.zeros_like(batch_inputs),batch_inputs)
            total_loss += loss
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.8f}")
        model.save(f"model_{epoch}_loss_{average_loss}")
