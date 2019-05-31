"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

import cv2
import numpy as np

from base import BaseOpticalFlow


class Dense(BaseOpticalFlow):

    def __init__(self):
        super().__init__()

    def process(self, previous_frame, next_frame, dimensions):
        """Computes the optical flow for all the points in the frame

        Args:

        Returns:

        """

        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # HSV - hue, saturation, value
        hsv = np.zeros(dimensions)

        # set saturation
        hsv[:, :, 1] = cv2.cvtColor(next_frame, cv2.COLOR_RGB2HSV)[:, :, 1]

        optical_flow = cv2.calcOpticalFlowFarneback(prev=previous_frame_gray, next=next_frame_gray, flow=None,
                                                    pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5,
                                                    poly_sigma=1.3, flags=0)

        # convert cartesian to polar
        magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

        # hue corresponds to direction
        hsv[:, :, 0] = angle * (180 / np.pi / 2)

        # value corresponds to magnitude - normalise between 0 and 255
        hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # convert HSV to int32's
        hsv = np.asarray(hsv, dtype=np.float32)

        # convert back to RGB for the network
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
