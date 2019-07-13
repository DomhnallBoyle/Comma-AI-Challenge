import argparse
import glob
import os
import pandas as pd
import re


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def main(args):
    with open(args.groundtruth_path, 'r') as f:
        groundtruth = [line.strip() for line in f]

    # sort frame locations alphanumerically
    frames = sorted_nicely(glob.glob(os.path.join(args.images_path, '*.jpg')))

    # should be same number of frames and groundtruth data
    assert len(groundtruth) == len(frames)

    df = pd.DataFrame({'frame': frames, 'speed': groundtruth})
    df.to_csv(os.path.join(args.output_directory, 'dataset.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images_path', type=str, help='Path to the training images')
    parser.add_argument('groundtruth_path', type=str, help='Path to the groundtruth')
    parser.add_argument('output_directory', type=str, help='Directory to output CSV to')

    main(parser.parse_args())
