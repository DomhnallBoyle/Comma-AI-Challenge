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
import shutil

from base import BaseOpticalFlow

DEBUG = False


class Dense(BaseOpticalFlow):

    def __init__(self, dimensions):
        super().__init__(dimensions=dimensions)

    def process(self, previous_frame, next_frame):
        """Computes the optical flow for all the points in the frame

        Args:

        Returns:

        """

        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # HSV - hue, saturation, value
        hsv = np.zeros(self.dimensions)

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


def main(args):
    """Take a video and run dense optical flow on each frame

    Outputs the results to disk

    :param args:
    :return:
    """

    if os.path.exists(args.output_directory):
        shutil.rmtree(args.output_directory)

    os.makedirs(args.output_directory)

    video_capture = cv2.VideoCapture(args.video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    number_of_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    previous_frame = video_capture.read()[1]

    dense_optical_flow = Dense(dimensions=(height, width, 3))

    frame_count = 1
    while video_capture.isOpened():
        if frame_count < number_of_frames:
            next_frame = video_capture.read()[1]

            optical_flow_output = dense_optical_flow.process(previous_frame=previous_frame, next_frame=next_frame)

            if args.debug:
                cv2.imshow('Previous', previous_frame)
                cv2.imshow('Next', next_frame)
                cv2.imshow('Optical Flow', optical_flow_output)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            previous_frame = next_frame

            cv2.imwrite(f'{args.output_directory}/{frame_count}.jpg', optical_flow_output)
            frame_count += 1
        else:
            break

    video_capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', type=str, help='Path to the frames to be converted')
    parser.add_argument('output_directory', type=str, help='Output directory of the new frames')
    parser.add_argument('--debug', type=bool, help='Debug mode', default=DEBUG)

    main(parser.parse_args())
