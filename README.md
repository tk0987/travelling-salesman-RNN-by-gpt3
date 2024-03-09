# travelling-salesman-RNN-by-gpt3

***
ATTENTION
***
created mainly by gpt3 from openai. under validation. currently - after firs epoch 2x times faster than connecting 3d points row/col-wise.
full validation soon.

older version:
epoch 51:

1/1 [==============================] - 2s 2s/step

(512, 1)

2.0881891 # this is how better NN performs than row/col-wise point connection. 16.02.2024 - previous, worse architecture. loss now updated, we will see what happen. bills are growing so.


my tf crashed after nvidia driver update (this one from january 2024). so, i have to use cpu so far.

***
REMEMBER - just logic. i want to say, that:
***

this network will never approach '0.0' loss. in training loop - every epoch - new dataset is drawn, with time seed. I hope that it will optimize its parameters, but i need to validate this first. hope is a mother of fool. we will see ;)
