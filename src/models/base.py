"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import random as rn
import shutil
import sys
from abc import ABC, abstractmethod
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import BatchCallback, DLBot, rmse, TelegramBotCallback

EPOCHS = 200
LEARNING_RATE = 1e-04
BATCH_SIZE = 40
TRAINING_TEST_SPLIT = 0.2
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', None)

# setting the seed environment variable
os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(2019)

# Setting the seed for python random numbers
# used to get the same training process each time if doing reruns
rn.seed(2019)


class BaseModel(ABC):

    def __init__(self, model_name, dimensions):
        # TODO: Data Augmentation Object
        self.model_name = model_name
        self.height, self.width, self.channels = dimensions
        self.parser = self.build_parser()
        self.args = self.parser.parse_args()
        self.batch_callback = BatchCallback()
        self.model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), self.model_name)

        if self.args.run_type in ['train', 'test']:
            getattr(self, self.args.run_type)()
        else:
            self.parser.print_help()

    def build_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='run_type', help='Sub-command help')

        training_parser = subparsers.add_parser('train', help='Training help')
        training_parser.add_argument('dataset_path', help='Absolute path to the dataset', type=str)
        training_parser.add_argument('--model_path', help='Path of the model to continue training', type=str)
        training_parser.add_argument('--train_test_split', help='Size of training/test split', type=float,
                                     default=TRAINING_TEST_SPLIT)
        training_parser.add_argument('--epochs', help='Number of epochs', type=int, default=EPOCHS)
        training_parser.add_argument('--batch_size', help='Size of the batches', type=int, default=BATCH_SIZE)
        training_parser.add_argument('--learning_rate', help='Learning rate', type=float, default=LEARNING_RATE)
        training_parser.add_argument('--crop_to', help='Area of frame to crop', type=int, default=0)

        testing_parser = subparsers.add_parser('test', help='Testing help')
        testing_parser.add_argument('dataset_path', help='Absolute path to the dataset', type=str)
        testing_parser.add_argument('model_path', help='Path to the model weights to load', type=str)
        testing_parser.add_argument('--crop_to', help='Area of frame to crop', type=int, default=0)

        return parser

    def load_frame(self, frame_path):
        """Load the frame given an absolute frame path

        Args:
            frame_path (String): absolute path of the frame

        Returns:
            (List): 3D list representing the read RGB frame
        """
        if not os.path.exists(frame_path):
            raise OSError('frame not found: ' + frame_path)

        return cv2.imread(frame_path, 1)

    def load_dataset(self, path):
        return pd.read_csv(path)

    def train(self):
        # TODO:
        #  Continue training
        if os.path.exists(self.model_directory):
            shutil.rmtree(self.model_directory)
        os.makedirs(self.model_directory)

        dataset = self.load_dataset(self.args.dataset_path)
        x, y = dataset['frame'].values, dataset['speed'].values

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.args.train_test_split,
                                                              random_state=0, shuffle=False)

        model = self.build_model()

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.args.learning_rate),
                      metrics=['mae', rmse])

        training_generator = self.batch_generator(x_train, y_train, training=True)
        validation_generator = self.batch_generator(x_valid, y_valid, training=False)

        # number of training and validation iterations/steps i.e. number of batches
        training_steps = int(len(x_train) / self.args.batch_size)
        validation_steps = int(len(x_valid) / self.args.batch_size)

        model.fit_generator(
            training_generator,
            steps_per_epoch=training_steps,
            epochs=self.args.epochs,
            max_queue_size=1,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=self.get_callbacks(),
            verbose=1
        )

    def test(self):
        dataset = self.load_dataset(self.args.dataset_path)
        frame_paths = dataset['frame'].values

        model = self.build_model()
        model.load_weights(self.args.model_path)

        predictions = []
        for frame_path in tqdm(frame_paths):
            frame = self.load_frame(frame_path=frame_path)
            frame = self.preprocess(frame=frame)
            frame = np.array([frame])  # model expects 4D
            predictions.append(model.predict(frame)[0][0])

        with open(f'{self.model_directory}/results.txt', 'w') as f:
            f.writelines([str(prediction) + '\n' for prediction in predictions])

    def batch_generator(self, frame_paths, speeds, training):
        batch_size = self.args.batch_size

        # batch size matrices for recording the data
        _frames = np.empty([batch_size, self.height, self.width, self.channels])
        _speeds = np.empty(batch_size)

        while True:
            # index for the matrices
            i = 0

            # reset indexes for next batch
            if training:
                index_start = self.batch_callback.training_sample_index
                index_end = self.batch_callback.training_sample_index + batch_size
            else:
                index_start = self.batch_callback.validation_sample_index
                index_end = self.batch_callback.validation_sample_index + batch_size

            # shuffle at beginning of an epoch if training so different variations of batches are chosen
            if training and self.batch_callback.begin_epoch:
                permutation = np.random.permutation(len(frame_paths))
                frame_paths = np.asarray(frame_paths)[permutation]
                speeds = np.asarray(speeds)[permutation]
                self.batch_callback.begin_epoch = False

            # load batches
            for index in (index_start, index_end):
                try:
                    frame_path, speed = frame_paths[index], float(speeds[index])

                    frame = self.load_frame(frame_path=frame_path)

                    _frames[i] = self.preprocess(frame=frame)
                    _speeds[i] = speed

                    i += 1
                except IndexError:
                    break

            self.batch_callback.training_sample_index += batch_size
            self.batch_callback.validation_sample_index += batch_size

            yield _frames, _speeds

    def get_callbacks(self):
        callbacks = []

        if TELEGRAM_TOKEN:
            bot = DLBot(token=TELEGRAM_TOKEN, user_id=None)
            telegram_callback = TelegramBotCallback(bot)
            callbacks.append(telegram_callback)
        else:
            # no telegram token environment variable
            print('No Telegram Token - not using Telegram callback')

        # callback to save the model after every 5 epochs
        checkpoint_path = os.path.join(self.model_directory, 'model-{epoch:03d}.h5')

        # save model after every 5 epochs
        checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, period=5)

        callbacks.extend([self.batch_callback, checkpoint_callback])

        return callbacks

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, **kwargs):
        raise NotImplementedError
