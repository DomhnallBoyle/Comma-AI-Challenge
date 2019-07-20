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

from base import BaseOpticalFlow

# feature default params
MAX_CORNERS = 100
QUALITY_LEVEL = 0.3
MIN_DISTANCE = 7
BLOCK_SIZE = 7

# optical flow default params
WIN_SIZE = (15, 15)
MAX_LEVEL = 2
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


class Sparse(BaseOpticalFlow):

    def __init__(self, dimensions, max_corners=MAX_CORNERS, quality_level=QUALITY_LEVEL, min_distance=MIN_DISTANCE,
                 block_size=BLOCK_SIZE, window_size=WIN_SIZE, max_level=MAX_LEVEL, criteria=CRITERIA):
        super().__init__(dimensions=dimensions)
        self.feature_params = {
            'maxCorners': max_corners,
            'qualityLevel': quality_level,
            'minDistance': min_distance,
            'blockSize': block_size
        }
        self.optical_flow_params = {
            'winSize': window_size,
            'maxLevel': max_level,
            'criteria': criteria
        }

    def process(self, previous_frame, next_frame, mask=None, previous_features=None):
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if previous_features is None:
            previous_features = cv2.goodFeaturesToTrack(previous_frame_gray, mask=None, **self.feature_params)

        if mask is None:
            # create mask for drawing purposes
            mask = np.zeros_like(previous_frame)

        next_features, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, next_frame_gray, previous_features, None,
                                                          **self.optical_flow_params)

        # select the good points
        good_new = next_features[st == 1]
        good_old = previous_features[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 2)
            next_frame = cv2.circle(next_frame, (a, b), 5, (0, 255, 0), -1)

        return cv2.add(next_frame, mask), mask, good_new.reshape(-1, 1, 2)


def main(args):
    video_capture = cv2.VideoCapture(args.video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sparse = Sparse(dimensions=(height, width, 3))

    previous_frame = video_capture.read()[1]
    previous_features = None
    mask = None

    while video_capture.isOpened():
        next_frame = video_capture.read()[1]

        optical_flow_output, mask, previous_features = sparse.process(previous_frame, next_frame, mask,
                                                                      previous_features)

        cv2.imshow('Optical Flow', optical_flow_output)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        previous_frame = next_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)

    args = parser.parse_args()

    main(args)
