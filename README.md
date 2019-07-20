Welcome to the comma.ai 2017 Programming Challenge!

Basically, your goal is to predict the speed of a car from a video.

data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
data/train.txt contains the speed of the car at each frame, one speed on each line.

data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
Your deliverable is test.txt

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

Usage: 

```
# get frames from the videos
python src/utils/video_to_frames.py data/train.mp4 data/train_images
python src/utils/video_to_frames.py data/test.mp4 data/test_images

# convert video to optical flow frames
python src/optical_flow/dense.py data/train.mp4 data/optical_flow_train
python src/optical_flow/dense.py data/test.mp4 data/optical_flow_test

# create training dataset
python src/utils/video_to_dataset.py data/train_images data/train.txt data

# train models
python src/models/nvidia.py train data/training_dataset.csv

# test models
python src/models/nvidia.py test data/testing_dataset.csv src/models/nvidia/model.h5
```