import tensorflow as tf
import numpy as np
import os
# import keras
# os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
@tf.function
def tsp_loss(y_true, y_pred):
    y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
    loss = tf.keras.losses.huber(y_true,y_pred[:])
    return tf.reduce_mean(loss)
    # return loss
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
    rd5,_,asdf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
    rd6,_,asdf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)
    rd7,_,asdf=tf.keras.layers.LSTM(n,return_sequences=True,return_state=True,go_backwards=True)(inputs)

    sum1=tf.keras.layers.Add()([rd1,rd2,rd3,rd4,rd5,rd6,rd7])
    sum2=tf.keras.layers.Add()([nd1,nd2,nd3,nd4,nd5,nd6,nd7])

    middle1=tf.keras.layers.Dense(num_cities,"elu")(sum1)
    middle2=tf.keras.layers.Dense(num_cities,"relu")(sum2)

    middle=tf.keras.layers.Add()([middle1,middle2])
    # middle=tf.keras.layers.Flatten()(middle)
    outputs=tf.keras.layers.Dense(1,"softmax")(middle)
    # print(tf.shape)
    return tf.keras.Model(inputs, outputs)

global num_cities
num_cities = 512

np.random.seed(99999)
# /Ubuntu/home/geniusz/nn/
# model = model1(11,num_cities=num_cities)
with tf.keras.utils.custom_object_scope({'tsp_loss': tsp_loss}):
    model = tf.keras.models.load_model("./model_0_loss_254003.09375")
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
sum_inp=np.sum(np.sqrt(inputs**2))
sum_preds=0.0
# print(len(sorted_inds[0]))
# print(len(inputs[0]))
# for asd in range(len(inputs)):
#     # for dsa in range(len(inputs[0])):
#     sum_inp+=np.sqrt(inputs[asd,0]**2+inputs[asd,1]**2+inputs[asd,2]**2)

for iii in range(len(inputs)):
    indd = sorted_inds[iii]
    print(np.shape(indd))
    # Use [0] to access the first (and only) element of the sorted indices
    x = inputs[indd, 0]
    y = inputs[indd, 1]
    z = inputs[indd, 2]
    # print(x,y,z)

    sum_preds += np.sqrt(x**2 + y**2 + z**2)

print(np.sum(sum_preds)/np.sum(sum_inp))
