# travelling-salesman-RNN-by-gpt3
i asked chatgpt for creating a rnn for tsp optimization (for my cnc). 
training loop is then an intelectual property of... gpt3. 
architecture is mine. quite a good, at least for being a scrap architecture. Shown here, as it needs more work. i will check for fully conv recurrent layers in keras, and if i find them - another source will be trashed here.

at least 18x better than connecting 3D points 'row_wise'. 
optimization for cnc still needed - this network does not accept that milled object have its borders.

loss value is very, very big. even after 100 epochs. but still - it works.

i won't upload saved model, because it's binary - i have my concerns 'bout anything binary
