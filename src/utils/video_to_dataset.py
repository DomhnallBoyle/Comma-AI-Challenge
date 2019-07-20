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


def get_frames(path):
    return sorted_nicely(glob.glob(f'{path}/*.jpg'))


def main(args):
    if args.mode == 'training':
        with open(args.groundtruth_path, 'r') as f:
            groundtruth = [line.strip() for line in f]

        frames = get_frames(args.images_path)
        groundtruth = groundtruth[0:len(frames)]

        df = pd.DataFrame({'frame': frames, 'speed': groundtruth})
    elif args.mode == 'testing':
        frames = get_frames(args.images_path)
        df = pd.DataFrame({'frame': frames})
    else:
        print('Incorrect mode...use training or testing')
        return

    df.to_csv(os.path.join(args.output_directory, f'{args.mode}_dataset.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='mode', help='Training or testing')

    parser_training = subparsers.add_parser('training', help='Training help')
    parser_training.add_argument('images_path', type=str, help='Path to the training images')
    parser_training.add_argument('groundtruth_path', type=str, help='Path to the groundtruth')
    parser_training.add_argument('output_directory', type=str, help='Directory to output CSV to')

    parser_testing = subparsers.add_parser('testing', help='Testing help')
    parser_testing.add_argument('images_path', type=str, help='Path to the training images')
    parser_testing.add_argument('output_directory', type=str, help='Directory to output CSV to')

    main(parser.parse_args())
