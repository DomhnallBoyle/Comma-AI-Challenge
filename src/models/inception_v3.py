"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, Flatten
from keras.models import Input, Model

from base import BaseModel

HEIGHT, WIDTH, CHANNELS = (75, 200, 3)


class InceptionV3Model(BaseModel):

    def __init__(self):
        super().__init__(height=HEIGHT, width=WIDTH, channels=CHANNELS)

    def build_model(self):
        """Overridden method for building the transfer learning InceptionV3 model structure

        Returns:
            model (Sequential): the model structure with the conv and dense layers as well as their settings
        """
        # download the InceptionV3 model weights based on ImageNet, don't include the last few layers
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH,
                                                                                                  CHANNELS)))

        # freeze all the layers - ensuring there are no changes to the weights of these
        for layer in base_model.layers:
            layer.trainable = False

        # construct the dense layers - similar to NVIDIA model
        # apply dropout, flatten and then dense layers with same activations
        x = base_model.output
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(units=100, activation='elu')(x)
        x = Dense(units=50, activation='elu')(x)
        x = Dense(units=10, activation='elu')(x)
        x = Dense(units=1, activation='elu')(x)

        # construct the model from the base model input and dense layers as output
        model = Model(inputs=base_model.input, outputs=x)

        # print the model summary
        model.summary()

        return model
