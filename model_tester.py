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
sorted_inds = np.argsort(preds,axis=1)
inputs=np.asanyarray(inputs)
# preds=np.asanyarray(preds)
sum_inp=np.sum(np.sqrt(inputs**2))
sum_preds=0.0
print(len(sorted_inds[0]))
print(len(inputs[0]))
# for asd in range(len(inputs)):
#     # for dsa in range(len(inputs[0])):
#     sum_inp+=np.sqrt(inputs[asd,0]**2+inputs[asd,1]**2+inputs[asd,2]**2)

for iii in range(len(inputs)):
    indd = sorted_inds[iii]

    # Use [0] to access the first (and only) element of the sorted indices
    x = inputs[indd[0], 0]
    y = inputs[indd[0], 1]
    z = inputs[indd[0], 2]

    sum_preds += np.sqrt(x**2 + y**2 + z**2)

print(np.sum(sum_preds)/np.sum(sum_inp))
