"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

from keras import backend as K


def rmse(y_true, y_pred):
    """Calculate the RMSE metric for keras models during training
    Args:
        y_true (List): list of groundtruth steering angles
        y_pred (List): list of predicted steering angles
    Returns:
        (Float): the RMSE
    """
    # root mean squared error (rmse) for regression
    # axis=-1
    # print(K.int_shape(y_pred))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))
