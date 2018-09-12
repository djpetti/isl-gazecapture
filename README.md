# ISL Mobile Gaze Experiment

This is a replication of the work published
[here.](http://gazecapture.csail.mit.edu/cvpr2016_gazecapture.pdf), done by the
RPI [Intelligent Systems Lab](https://www.ecse.rpi.edu/~cvrl/).

The goal of this research is to produce a practical system for tracking the 2D
position of the user's gaze on a mobile device screen. It uses an
appearance-based method that employs a CNN operating on images of the user's
eyes and face taken by the device's front-facing camera. Nominally, it can
achieve ~2 cm net accuracy, as mesured on the GazeCapture validation dataset.

# Using this Code

The code available here can be used for two basic functions. The first is
constructing and training the CNN model. The second is running a simple demo
using an already-trained model. The demo works by creating a simple server that
can be passed images by the client, and send back the estimated gaze point.
There is also a demonstration Android app that acts as a client and implements a
simple demo which we intend to make available in the future.

## Training the Model

Training the model requires TensorFlow, Keras, and
[Rhodopsin](https://github.com/djpetti/rhodopsin). Practically, it also requires
a GPU, preferably one with at least 8 Gb of VRAM.

### Building the Dataset

Training the model requires the [GazeCapture](http://gazecapture.csail.mit.edu/)
dataset. The dataset must also be converted to a TFRecords format before it can
be used for training. Once the dataset has been downloaded, there is an included
script to perform this conversion:

```
~$ ./process_gazecap.py dataset_dir output_dir
```

The first argument is the path to the root directory of the GazeCapture dataset
that you downloaded. The second argument is the path to the output directory
where you want the TFRecords files to be located.

This script will create three TFRecords files in the output directory: One for
the training data, one for the testing data, and one for the validation data.

For convenience, this script can be run in the same Docker container as the
actual training. (See below.)

### Performing the Training

By far, the easiest way to train the model is to use the pre-built [Docker
container](https://hub.docker.com/r/djpetti/isl-gazecapture/). There is a script
included that will automatically pull this container and open an interactive
shell:

```
~$ ./start_train_container.sh
```

Note that this requires both Docker and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to be installed on your
local machine.

The script automatically mount-binds the repository directory to the
`isl_gazecapture` directory inside the container. Once inside this directory,
training can be initiated as follows:

```
~$ ./train_gazecap.py train_dataset test_dataset
```

The first argument is the path to the TFRecords file containing the training
dataset. Likewise, the second argument is the path to the file containing the
testing dataset.

### Configuring Training

Many of the attributes that pertain to training are set in constants defined at
the top of the `train_gazecap.py` file. These can be modified at-will, and have
comments documenting their functions.

Additionally, settings that are common to both the training procedure and the
demo server are located in `itracker/common/config.py`. Most likely, the only
parameter here that you might want to modify is `NET_ARCH`. This parameter
points to the class of the network to train. (Different classes are defined in
`network.py` for different network architectures.) This can be changed to any of
the classes specified in `network.py` in order to experiment with alternative
architectures.

### Training a Saved Model

You can start training with a pre-trained model if you wish. This is done by
passing the hd5 file containing the saved model weights with the `-m` parameter
to the training script.

Note that the architecture of saved models cannot be automatically detected, so
it must be ensured that the current value of `NET_ARCH` matches the saved model
architecture.

### Validating a Trained Model

You may with to evaluate the accuracy of the model on the GazeCapture validation
dataset. This dataset should be automatically generated, but is not used during
the training procedure.

In order to validate a saved model, specify the location of the validation
dataset using the `-v` option. (Note that you will also have to specify the saved
model with the `-m` option for this to work.)

## Running the Demo

TODO (djpetti): Write this section.
