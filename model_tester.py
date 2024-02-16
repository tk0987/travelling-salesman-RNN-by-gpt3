import tensorflow as tf
import numpy as np
import os
with tf.device('/device:CPU:0'): # i cannot do this another way, something is scr~wed after nvidia driver update
# import keras
# os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
    @tf.function
    def tsp_loss(y_true, y_pred):
        y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
        loss = tf.keras.losses.huber(y_true,y_pred[:])
        return tf.reduce_mean(loss)
        # return loss


    global num_cities
    num_cities = 512

    np.random.seed(99999)
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
        model = tf.keras.models.load_model("model_51_loss_254.0030975341797")
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
