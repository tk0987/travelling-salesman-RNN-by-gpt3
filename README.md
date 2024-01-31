# travelling-salesman-RNN-by-gpt3

***
ATTENTION
***
i have several versions of this code, while only one is correct. i'm checking this version currently, my pc is humming nicely. and my bills are growing nicely, too!

i asked chatgpt for creating a rnn for tsp optimization (for my cnc). 
training loop is then an intelectual property of... gpt3. 
architecture is mine. quite a good, at least for being a scrap architecture. Shown here, as it needs more work. i will check for fully conv recurrent layers in keras, and if i find them - another source will be trashed here.

at least 2x better than connecting 3D points 'row/column_wise', with a simply np.sum() ratio in cartesian coords. makes sense, makes 'being cautious'. the training loop needs to be improved (this np.random...)
optimization for cnc still needed - this network does not accept that milled object have its borders.

loss value is very, very big. even after 100 epochs. but still - it works.

i won't upload saved model, because it's binary - i have my concerns 'bout anything binary

at 1st feb 2024 i 'll try this thing on real gcode, in order to confirm this non-confirmed results


as stands for 31'th jan 2024 - it wont be better. maybe tommorow (i've achieved 9x better result with dense layers). maybe it won't be better at all. no matter, it is time to be cautious 'bout myself, at least for me.
