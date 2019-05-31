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


MAX_CORNERS = 100
QUALITY_LEVEL = 0.3
MIN_DISTANCE = 7
BLOCK_SIZE = 7


def dense(next_frame, previous_frame, dimensions):
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


def sparse(next_image, previous_image, **kwargs):
    """Lucas Kanade method computes optical flow at well detected image references.

    It finds a good point in image_current (pt1), and then it find where the same point translated to in image_next
    (pt2). Then it computes the optical flow from pt1 to pt 2

    Args:

    """

    feature_params = dict(maxCorners=kwargs['max_corners'], qualityLevel=0.3, minDistance=7, blockSize=7)


def video_pipeline(video_path, debug=False):
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_frame = video_capture.read()[1]

    while video_capture.isOpened():
        next_frame = video_capture.read()[1]

        if debug:
            cv2.imshow('Previous', previous_frame)
            cv2.imshow('Next', next_frame)

            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

        # optical_flow_output = dense_optical_flow(next_frame, previous_frame, dimensions=(height, width, 3))

        cv2.imshow('Optical Flow', optical_flow_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame = next_frame


def main(args):
    video_pipeline(args.video_path, debug=args.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', type=str, help='Path to the video to perform optical flow.')
    parser.add_argument('--type', type=str, help='dense/sparse optical flow', default='dense')
    parser.add_argument('--debug', type=bool, help='Debug mode', default=False)

    main(parser.parse_args())
