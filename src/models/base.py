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
import os
from abc import ABC, abstractmethod

EPOCHS = 200
LEARNING_RATE = 1e-04
BATCH_SIZE = 40
TRAINING_TEST_SPLIT = 0.2
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', None)


class BaseModel(ABC):

    def __init__(self, dimensions):
        # TODO: Data Augmentation Object
        self.dimensions = dimensions
        self.parser = self.build_parser()
        self.args = self.parser.parse_args()

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

        testing_parser = subparsers.add_parser('test', help='Testing help')

        return parser

    def load_image(self, image_path):
        """Load the image given an absolute image path

        Args:
            image_path (String): absolute path of the image

        Returns:
            (List): 3D list representing the read RGB image
        """
        if not os.path.exists(image_path):
            raise OSError('Image not found: ' + image_path)

        return cv2.imread(image_path, 1)

    def load_training_data(self):
        dataset_directory = self.args.dataset_path



    def train(self):
        pass

    def test(self):
        pass

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError
