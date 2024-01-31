import tensorflow as tf
import numpy as np

@tf.function
def tsp_loss(y_true, y_pred):
    y_true = tf.cast(tf.sparse.to_dense(y_true), dtype=tf.float32)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return tf.reduce_mean(loss)
num_cities = 1000
np.random.seed(999)
# /Ubuntu/home/geniusz/nn/
model = tf.keras.models.load_model(f"./model_229_loss_14669.806640625.h5",custom_objects={'tsp_loss': tsp_loss})
# model.compile()
# model.load_weights(f"/geniusz/nn/tsp/my_recurrent/variables/variables.data-00000-of-00001")
def generate_tsp_data(num_cities):
    np.random.seed(0)
    return np.random.rand(num_cities, 3)
num_samples = 10
inputs = np.expand_dims(generate_tsp_data(num_cities),axis=0) # bah, abrakadabra! 
preds=[]
# for i in range(inputs[0]):
preds=(model.predict(inputs))

inputs=np.asanyarray(inputs)
preds=np.asanyarray(preds)
print((np.sum(np.sqrt(inputs[:,:,0]**2+inputs[:,:,1]**2+inputs[:,:,2]**2))/np.sum(np.sqrt(preds[:,:,0]**2+preds[:,:,1]**2+preds[:,:,2]**2)))**-1)# this '**-1' is because i'm lazy today
