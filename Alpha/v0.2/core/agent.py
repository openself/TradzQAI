import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, PReLU, CuDNNLSTM, Flatten, Reshape, Dropout
from keras.optimizers import Adam

import os
import time

import numpy as np
import random
from collections import deque

from core.environnement import *

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        columns = ['Price', 'POS', 'Order']
        self.inventory = pd.DataFrame(columns=columns)
        self.mode = ""
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        if os.path.exists("models/" + model_name):
            print ("Model loaded")
            self.model = load_model("models/" + model_name)
        else:
            print ("Model builded")
            self.model = self._model([2, 1, ])

    def _model(self, layers):
        d = 0.2
        model = Sequential()
        model.add(Reshape((1, 2), input_shape=(2, )))
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(d))
        model.add(PReLU())
        model.add(CuDNNLSTM(64, return_sequences=False))
        model.add(Dropout(d))
        model.add(PReLU())
        model.add(Dense(32))
        model.add(PReLU())
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
