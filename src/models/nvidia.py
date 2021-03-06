"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

import cv2
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential

from base import BaseModel

HEIGHT, WIDTH, CHANNELS = (66, 200, 3)


class NvidiaModel(BaseModel):
    """

    Based on:
    https://arxiv.org/abs/1604.07316

    """

    def __init__(self):
        super().__init__(model_name='nvidia', dimensions=(HEIGHT, WIDTH, CHANNELS))

    def build_model(self):
        model = Sequential()

        # image normalisation layer - to avoid saturation and make gradients work better
        # CNN performs optimally when working when small, floating point values are processed
        # appearance of the images are unaltered after this step
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(HEIGHT, WIDTH, CHANNELS),
                         output_shape=(HEIGHT, WIDTH, CHANNELS)))

        # 5 convolutional layers
        # uses valid padding - output maps are smaller than the input
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))

        # dropout & flatten
        # dropout - randomly selected neurons are ignored during training. Their activation is temporally removed on
        # the forward pass and any weight updates are not applied on the backward pass. Other neurons will have to step
        # in and handle the representation required to make predictions for the missing neurons. The network then
        # becomes less sensitive to the specific weights of neurons resulting in better generalisation and less likely
        # to overfit
        model.add(Dropout(rate=0.5))
        model.add(Flatten())

        # dense layers
        model.add(Dense(units=100, activation='elu'))
        model.add(Dense(units=50, activation='elu'))
        model.add(Dense(units=10, activation='elu'))
        model.add(Dense(units=1))

        # print summary of structure
        model.summary()

        return model

    def preprocess(self, **kwargs):
        frame = kwargs['frame']

        # crop
        frame = frame[self.args.crop_to:, :, :]

        # resize
        frame = cv2.resize(frame, (self.width, self.height), cv2.INTER_AREA)

        # convert to YUV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        return frame


if __name__ == '__main__':
    NvidiaModel()
