import tensorflow as tf
import numpy as np
import os
from datetime import datetime
with tf.device('/device:CPU:0'):
# import keras
# os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
    # @tf.function
    @tf.function
    def tsp_loss(y_true, y_pred):
        y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return tf.reduce_mean(loss)


    global num_cities
    num_cities = 1000

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

    seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        from tensorflow.python.keras import backend
        arg_0_tensor = tf.saturate_cast(tf.random.uniform([], minval=0, maxval=2, dtype=tf.int64), dtype=tf.uint64)
        arg_0 = tf.identity(arg_0_tensor)
        arg_1 = -674
        arg_2 = False
        out = backend.clip(arg_0,arg_1,arg_2)
    except Exception as e:
        print("Error:"+str(e))
    # /Ubuntu/home/geniusz/nn/
    # model = model1(11,num_cities=num_cities)
    with tf.keras.utils.custom_object_scope({'tsp_loss': tsp_loss}):
        model.load_weights("model_0_loss_331841.062500.weights.h5")
        # model.compile()
    # model.compile()
    # model.load_weights(f"/geniusz/nn/tsp/my_recurrent/variables/variables.data-00000-of-00001")
    def generate_tsp_data(num_cities):
        # np.random.seed(342141234)
        return np.random.rand(num_cities, 3)
    # num_samples = 10
    inputs = generate_tsp_data(num_cities) # bah, abrakadabra! 
    inputs=np.expand_dims(inputs,axis=0)
    inputs=tf.convert_to_tensor(inputs,dtype=tf.float32)
    preds=[]
    # for i in range(inputs[0]):
    preds=model.predict(inputs)
    # for el in preds:
    #     print(el)
    sorted_inds = np.argsort(preds,axis=-1)
    inputs=np.asanyarray(inputs)
    # preds=np.asanyarray(preds)
    sum_inp=np.sum(np.sqrt(inputs[0,:,0]**2+inputs[0,:,1]**2+inputs[0,:,2]**2))
    sum_preds=0.0
    # print(len(sorted_inds[0]))
    # print(len(inputs[0]))
    # for asd in range(len(inputs)):
    #     # for dsa in range(len(inputs[0])):
    #     sum_inp+=np.sqrt(inputs[asd,0]**2+inputs[asd,1]**2+inputs[asd,2]**2)

    for iii in range(len(sorted_inds[0])):
        indd = sorted_inds[0,iii]
        # print(np.shape(inputs))
        # Use [0] to access the first (and only) element of the sorted indices
        x = inputs[0,indd,0]
        y = inputs[0,indd,1]
        z = inputs[0,indd,2]
        # print(x,y,z)

        sum_preds += np.sqrt(x**2 + y**2 + z**2)

    print(np.sum(sum_preds)/np.sum(sum_inp))
