import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
with tf.device('/device:CPU:0'):
# import keras
# os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
    # @tf.function
    @tf.function
    def travelling_loss(y_true,y_pred):
        indices=tf.argsort(y_pred)
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true,inputs[indices]))


    global num_cities
    num_cities = 1024

    np.random.seed(int(datetime.now().timestamp()))
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
    with tf.keras.utils.custom_object_scope({'travelling_loss': travelling_loss}):
        model = tf.keras.models.load_model("model_2_loss_0.3357119560241699")
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
    sorted_inds = np.argsort(np.asanyarray(preds),axis=-1)
    print(np.shape(sorted_inds)," sorted inds shape")
    inputs=np.asanyarray(inputs)
    # preds=np.asanyarray(preds)
    # for asdf in range(len(np.asanyarray(inputs[0]))):
    #     for fdsafds in range(len(inputs[0,0])):
    sum_inp=np.sum(np.sqrt(inputs**2))

    
    sum_preds=0.0
    # print(len(sorted_inds[0]))
    # print(len(inputs[0]))
    # for asd in range(len(inputs)):
    #     # for dsa in range(len(inputs[0])):
    #     sum_inp+=np.sqrt(inputs[asd,0]**2+inputs[asd,1]**2+inputs[asd,2]**2)
    dummy=[]
    
    for iii in range(len(sorted_inds[0])):
        indd = sorted_inds[0,iii,0]
        # print(indd," index ",iii," badziew")
        # Use [0] to access the first (and only) element of the sorted indices
        x = inputs[0,indd, 0]
        y = inputs[0,indd, 1]
        z = inputs[0,indd, 2]

        # print(z," zet")
        # print(x,y,z)
        dummy.append([x,y,z])
        sum_preds += np.sqrt(np.sum(x**2 + y**2 + z**2))
    # inputs=np.asanyarray(inputs)
    print(preds)
    print(np.shape(inputs[0]),len(inputs[0]),np.shape(sorted_inds),len(sorted_inds[0]))
    print(np.shape(inputs)," inputs shape")

    print(np.sum(sum_preds)/np.sum(sum_inp))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(inputs[0,:,0],inputs[0,:,1],inputs[0,:,2],linewidth=0.07,c='b')

    plt.show()
    # dummy=np.asanyarray(dummy)
    print(np.shape(dummy)," dummy shape")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(dummy[:][0],dummy[:][1],dummy[:][2])

    plt.show()

    print([(dummy[0][:][0],dummy[0][:][1],dummy[0][:][2]),(inputs[0,:,0],inputs[0,:,1],inputs[0,:,2])])
