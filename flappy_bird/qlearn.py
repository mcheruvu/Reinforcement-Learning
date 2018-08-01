#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
#import wrapped_flappy_bird as game
from flappy_bird_env import * 
import random
import numpy as np
from collections import deque
import datetime
import json

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 1000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.08 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    max_episode=2000000
    env = FlappyBirdEnv()
    start = datetime.datetime.now()
    algorithm = 'DQN'
    data = []
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    #do_nothing = np.zeros(ACTIONS)
    #do_nothing[0] = 1

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    for t in range( max_episode):
        x_t = env.reset(return_type=3)
  
        x_t = skimage.color.rgb2gray(x_t)
        x_t = skimage.transform.resize(x_t,(80,80))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

        x_t = x_t / 255.0

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
        
        #loss = 0
        #Q_sa = 0
        terminal=False
        while not terminal:
            loss = 0
            Q_sa = 0
            #action_index = 0
            r_t = 0
            a_t = 0
        #choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action = random.randrange(ACTIONS)
                    a_t = action
                    #print("a_t", a_t)
                else:
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action = max_Q
                    a_t = action
                    #print("a_t", a_t)

        #We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
            x_t1_colored, r_t, terminal,_ = env.step(a_t,return_type=3)
            env.render()
            x_t1 = skimage.color.rgb2gray(x_t1_colored)
            x_t1 = skimage.transform.resize(x_t1,(80,80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))


            x_t1 = x_t1 / 255.0


            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
            D.append((s_t, action, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        #only train if done observing
            if t > OBSERVE:
            #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

            #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, done = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                #print ("Q_sa", Q_sa)
                #print ("Max Q_sa",np.max(Q_sa, axis=1))
                targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(done)

                loss += model.train_on_batch(state_t, targets)
            
     
            s_t = s_t1
            #if terminal:
            #    break
            
        t = t + 1
        
        
        duration = datetime.datetime.now() - start 
        
        if (env.score >= 10):
            print("Duration: {} Episode {} Score: {}".format(duration, 
                                                                t, 
                                                                env.score))
        
        data.append(json.dumps({ "algorithm": algorithm, 
                    "duration":  "{}".format(duration), 
                    "episode":   t, 
                    "reward":    r_t, 
                    "score":     env.score}))
        
        if (len(data) == 500):
            file_name = 'data/stats_flappy_bird_{}.json'.format(algorithm)
            
            # delete the old file before saving data for this session
            #if t == 1 and os.path.exists(file_name): os.remove(file_name)
                
            # open the file in append mode to add more json data
            file = open(file_name, 'a+')  
            for item in data:
                file.write(item)  
                file.write(",")
            #end for
            file.close()
            
            data = []
            
        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
