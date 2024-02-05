#update the architecture. working on 8gb rtx2080. kinda slow, as it still uses cpu mainly. much slower than previous version.
#different training loop. created with help of openai's gpt3 (this free sth). as the whole project was intended
#loss updated

import tensorflow as tf
import numpy as np
from datetime import datetime
import random as r
import keras
gpus = tf.config.experimental.list_physical_devices('GPU')
seed=datetime.now().timestamp()
def uniform_0to1_gen():
    rand_max = 1e30

    uniform = r.randint(0,rand_max)/(1+rand_max)
    return uniform
def uniform_XtoY_gen(x,y):
    rand_max = 1e30

    uniform_xy = (y-x)*(r.randint(0,rand_max)/(1+rand_max))+x
    return uniform_xy
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# Set the random seeds for reproducibility
with tf.device('/device:GPU:0'):
    seed = 777
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Generate random 3D TSP data
    def generate_tsp_data(num_cities):
        np.random.seed(seed)
        return 100*np.random.rand(num_cities, 3) # according to gcode in [mm], it is a cube 100x100x100 mm^3 or mm**3

    @tf.function
    def tsp_loss(y_true, y_pred):
        # Ensure y_true and y_pred have the same dtype
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Compute the mean squared error
        loss = keras.losses.mean_squared_error(y_true, y_pred)#updated, working, not validated yet

        return loss







    # Create the model - it is a scrap, but working scrap - better than connecting points 'row_wise'
    # kinda crude, and scrap - beware
    num_cities=2000
    
    def model1(n,num_cities): # i hope that it will be better than connecting points col_wise/row_wise
        inputs=tf.keras.layers.Input(shape=(num_cities, 3), batch_size=1)
        nd1,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd2,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd3,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd4,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd5,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd6,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        nd7,_=tf.keras.layers.SimpleRNN(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

        rd1,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd2,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd3,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd4,_=tf.keras.layers.GRU(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd5,_,asdf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd6,_,asdf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
        rd7,_,asdf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

        sum1=tf.keras.layers.Add()([rd1,rd2,rd3,rd4,rd5,rd6,rd7])
        sum2=tf.keras.layers.Add()([nd1,nd2,nd3,nd4,nd5,nd6,nd7])

        middle1=tf.keras.layers.Dense(2*n,"elu")(sum1)
        middle2=tf.keras.layers.Dense(2*n,"elu")(sum2)

        middle=tf.keras.layers.Add()([middle1,middle2])
        middle=tf.keras.layers.Flatten()(middle)
        outputs=tf.keras.layers.Dense(1,"softmax")(middle)

        return tf.keras.Model(inputs,outputs)

    # tsp_data = generate_tsp_data(num_cities)
    
    model=model1(2,num_cities)
    model.summary()
    # Compile the model
    optimizer = tf.keras.optimizers.AdamW(0.0005,0.007,0.8,0.98,1e-6)
    model.compile(optimizer=optimizer, loss=tsp_loss)


    # Define the training step
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
        permutation = np.random.permutation(num_cities)
        tsp_data = np.asanyarray(generate_tsp_data(num_cities))
        inputs = np.asanyarray(tsp_data[permutation])
        outputs_indices = np.arange((num_cities))
        outputs_values = np.zeros((num_cities, 3), dtype=np.int32)
        batch_size = 1
        inputs=np.expand_dims(inputs,axis=0)

        outputs_indices = outputs_indices.astype(np.float32)  # Convert to int8 data type
        outputs_values = outputs_values.astype(np.float32)  # Convert to int8 data type

        loss = train_step(tf.convert_to_tensor(inputs,dtype=tf.float32), tf.convert_to_tensor(outputs_indices))

        print(f"\n\n\nEpoch {epoch + 1}/{num_epochs}\n\n\n")

        model.save(f"./model_{epoch}_loss_{tf.reduce_sum(loss)}.h5", overwrite=False)
