import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
from utils import dotdict


class NNetWrapper:

    def __init__(self, game):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.dropout = 0.3
        self.epochs = 5
        self.batch_size = 64
        self.num_channels = 256

        inputs = keras.Input(shape=(self.board_y, self.board_x))  # Returns a placeholder tensor

        action_size = game.getActionSize()

        x = keras.layers.Reshape((self.board_y, self.board_x, 1))(inputs)

        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)

        x = keras.layers.Reshape((self.board_y*self.board_x*self.num_channels,))(x)

        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)

        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)

        pi = keras.layers.Dense(action_size, activation='softmax', name='pi')(x)

        v = keras.layers.Dense(3, activation='relu', name='v')(x)

        self.model = keras.Model(inputs=inputs, outputs=[pi, v])
        self.model.compile(optimizer='adam',
                      loss={'pi': 'categorical_crossentropy',
                          'v': 'mean_squared_error'},
                      metrics=['accuracy'])

    def res_block(self, input, filter_size):
        x = keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([input, x])
        x = keras.layers.ReLU()(x)
        return x

    def train(self, examples):
        boards, pis, vs = list(zip(*examples))
        self.model.fit(np.array(boards), [np.array(pis), np.array(vs)], epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, board, player):
        board = board[np.newaxis, :, :]
        board = board.astype('float32')
        [pi, v] = self.model.predict(board)
        pi = np.reshape(pi, (self.action_size,))
        v = np.reshape(v, (3,))
        if player == 2:
            v = np.array([v[2], v[0], v[1]])
        elif player == 3:
            v = np.array([v[1], v[2], v[0]])

        return pi, v

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        self.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath+'.meta'):
        #     raise("No model in path {}".format(filepath))
        self.model = keras.models.load_model(filepath)
