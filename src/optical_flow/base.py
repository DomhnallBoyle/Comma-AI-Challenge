"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

from abc import ABC, abstractmethod


class BaseOpticalFlow(ABC):

    def __init__(self, dimensions):
        self.dimensions = dimensions  # height, width, channels

    @abstractmethod
    def process(self, previous_frame, next_frame, **kwargs):
        raise NotImplementedError
