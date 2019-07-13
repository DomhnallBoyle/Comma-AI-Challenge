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
import shutil


def main(args):
    """

    :param args:
    :return:
    """
    video_capture = cv2.VideoCapture(args.video_path)
    number_of_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    output_directory = args.output_directory

    # remove output folder if exists
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    while video_capture.isOpened():
        success, frame = video_capture.read()

        if success:
            frame_count += 1
            image_filename = os.path.join(output_directory, '{}.jpg'.format(frame_count))
            cv2.imwrite(image_filename, frame)

            if args.debug:
                cv2.imshow('Image', frame)
                cv2.waitKey(1)
        else:
            break

    try:
        assert number_of_frames == frame_count
    except AssertionError:
        print('Number of frames: {}'.format(number_of_frames))
        print('Frame count: {}'.format(frame_count))

    cv2.destroyAllWindows()
    video_capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', type=str, help='Path to the video')
    parser.add_argument('output_directory', type=str, help='Absolute directory to save the images to')
    parser.add_argument('--debug', type=bool, help='Debug mode', default=False)

    main(parser.parse_args())
