# starting_guide_pygames_AI
Starting guild for Pygame AI training and play

<p align="center" width="100%">
    <img width="25%" src="https://github.com/jkaewprateep/Python_and_CURL/blob/main/Python.jpg">
    <img width="24%" src="https://github.com/jkaewprateep/starting_guide_pygames_AI/blob/main/pygame.jpg">
    <img width="18%" src="https://github.com/jkaewprateep/starting_guide_pygames_AI/blob/main/image10.jpg">
    <img width="12%" src="https://github.com/jkaewprateep/starting_guide_pygames_AI/blob/main/image6.jpg"> </br>
    <b> Pygame and Tensorflow AI machine learning </b> </br>
    <b> ( Picture from Internet ) </b> </br>
</p>

[Flappybird games]( https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html#rewards )

<p align="center" width="100%">
    <img width="25%" src="https://github.com/jkaewprateep/starting_guide_pygames_AI/blob/main/2000%20hits-training.gif"></br>
    <b> Training 2,000 times action by guilding algorithms </b> </br>
</p>
</br>
</br>

<p align="center" width="100%">
    <img width="25%" src="https://github.com/jkaewprateep/starting_guide_pygames_AI/blob/main/FlappyBird_small.gif"></br>
    <b> Training more than 10,000 times action by guilding algorithms </b> </br>
</p>
</br>
</br>

🧸💬 Because my machine is slow response ( my sister's University Laptop ) and screen size cannot fit the game resolution, use of the command ```os.environ['SDL_VIDEO_WINDOW_POS']``` to set PyGame environment requirements. </br>
```
x = 100
y = 100
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

import ple
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w, K_a, K_d, K_s, K_h 

import os
from os.path import exists

import tensorflow as tf
```

🧸💬 Learning parameters  </br>
```
learning_rate = 0.0001
momentum = 0.4
batch_size=1
```

🧸💬 Guidling function 1
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def random_action( ): 
	
	temp = tf.random.normal([1, 2], 0.2, 0.8, tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ 0.3, 0.7 ], shape=(2, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action
```

🧸💬 Prediction function
```
def predict_action( DATA ):
	
	# temp = DATA[0,:,:,:,:]
	# print( temp.shape )

	predictions = model.predict(DATA)
	score = tf.nn.softmax(predictions[0])

	return int(tf.math.argmax(score))
```

🧸💬 Model requirements
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (5, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    
    tf.keras.layers.Reshape((1, 5)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))

])
        
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredError(
    reduction='sum_over_batch_size',
    name='mean_squared_error'
)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])
```

🧸💬 Game environment
```
game = FlappyBird(width=216, height=384)
p = PLE(game, fps=30, display_screen=True
actions = { "none_1": None, "up___1": K_w };

p.init()
reward = 0.0
nb_frames = 1000000000
```

🧸💬 Model training requirements
```
history = [];

checkpoint_path = "C:\\temp\\python\\models\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)
 
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
```

🧸💬 Game play and model training
```
for i in range(nb_frames):
    
    if p.game_over():
        # input('pause')
        p.reset_game()    
    
    ## data row
    _gamestate = p.getGameState();

    DATA = tf.constant([ _gamestate["next_pipe_dist_to_player"], _gamestate["player_y"],
                            int(_gamestate["next_pipe_bottom_y"]), reward, i ]
                       , shape=(1,1,5,1))
    LABEL = tf.constant([random_action()], shape=(1,1,1)) 

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : DataSet
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    : Training
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # history = model.fit(dataset, epochs=3, callbacks=[custom_callback])
    history = model.fit(dataset, epochs=3)
    model.save_weights(checkpoint_path)

    action = predict_action(DATA);
    
    if i < 2000 : # for training guild function 2
        if _gamestate["player_y"] - int(_gamestate["next_pipe_bottom_y"]) > -35 :
            action = 1;
        else :
            action = 0;
        
    if i % 10000 == 0 :
        i == 0;

    reward = p.act(list(actions.values())[action])
    
    str_mode = "AI machine"; # game play mode
    if i < 2000 :
        str_mode = "guidling." # game play mode
    print("=============================================================================== \
                    ================================>", str_mode);
```
