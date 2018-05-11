from keras.models import Model, load_model
import keras.backend as K
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers

import tensorflow as tf


# Shape of the inputs to the network.
INPUT_SHAPE = (224, 224, 3)


class Network(object):
  """ Represents a network. """

  def __init__(self, fine_tune=False, data_tensors=None):
    """ Creates a new network.
    Args:
      fine_tune: Whether we are fine-tuning the model.
      data_tensors: If specified, the set of output tensors from the pipeline,
                    which will be used to build the model. """
    self.__data_tensors = data_tensors
    self._fine_tune = fine_tune

  def _build_common(self):
    """ Build the network components that are common to all. """
    # L2 regularizer for weight decay.
    self._l2 = regularizers.l2(0.0005)

    leye = None
    reye = None
    face = None
    grid = None
    if self.__data_tensors:
      leye, reye, face, grid = self.__data_tensors

    # Create inputs.
    self._left_eye_input = layers.Input(shape=INPUT_SHAPE, tensor=leye,
                                        name="left_eye_input")
    self._right_eye_input = layers.Input(shape=INPUT_SHAPE, tensor=reye,
                                         name="right_eye_input")
    self._face_input = layers.Input(shape=INPUT_SHAPE, tensor=face,
                                    name="face_input")
    self._grid_input = layers.Input(shape=(25, 25), tensor=grid,
                                    name="grid_input")

  def _build_custom(self):
    """ Builds the custom part of the network. Override this in a subclass.
    Returns:
      The outputs that will be used in the model. """
    raise NotImplementedError("Must be implemented by subclass.")

  def build(self):
    """ Builds the network.
    Returns:
      The built model. """
    # Build the common parts.
    self._build_common()
    # Build the custom parts.
    outputs = self._build_custom()

    # Create the model.
    model = Model(inputs=[self._left_eye_input, self._right_eye_input,
                          self._face_input, self._grid_input],
                  outputs=outputs)
    model.summary()

    return model

class MitNetwork(Network):
  """ This is the standard network architecture, based off of the one described
  in the Gazecapture paper. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Shared eye layers.
    conv_e1 = layers.Conv2D(96, (11, 11), strides=(4, 4), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e1 = layers.BatchNormalization(trainable=trainable)

    pad_e2 = layers.ZeroPadding2D(padding=(2, 2))
    conv_e2 = layers.Conv2D(256, (5, 5), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e2 = layers.BatchNormalization(trainable=trainable)

    pad_e3 = layers.ZeroPadding2D(padding=(1, 1))
    conv_e3 = layers.Conv2D(384, (3, 3), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)

    conv_e4 = layers.Conv2D(64, (1, 1), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    flatten_e4 = layers.Flatten()

    # Left eye stack.
    leye_conv_e1 = conv_e1(self._left_eye_input)
    leye_pool_e1 = pool_e1(leye_conv_e1)
    leye_norm_e1 = norm_e1(leye_pool_e1)

    leye_pad_e2 = pad_e2(leye_norm_e1)
    leye_conv_e2 = conv_e2(leye_pad_e2)
    leye_pool_e2 = pool_e2(leye_conv_e2)
    leye_norm_e2 = norm_e2(leye_pool_e2)

    leye_pad_e3 = pad_e3(leye_norm_e2)
    leye_conv_e3 = conv_e3(leye_pad_e3)

    leye_conv_e4 = conv_e4(leye_conv_e3)
    leye_flatten_e4 = flatten_e4(leye_conv_e4)

    # Right eye stack.
    reye_conv_e1 = conv_e1(self._right_eye_input)
    reye_pool_e1 = pool_e1(reye_conv_e1)
    reye_norm_e1 = norm_e1(reye_pool_e1)

    reye_pad_e2 = pad_e2(reye_norm_e1)
    reye_conv_e2 = conv_e2(reye_pad_e2)
    reye_pool_e2 = pool_e2(reye_conv_e2)
    reye_norm_e2 = norm_e2(reye_pool_e2)

    reye_pad_e3 = pad_e3(reye_norm_e2)
    reye_conv_e3 = conv_e3(reye_pad_e3)

    reye_conv_e4 = conv_e4(reye_conv_e3)
    reye_flatten_e4 = flatten_e4(reye_conv_e4)

    # Concatenate eyes and put through a shared FC layer.
    eye_combined = layers.Concatenate()([reye_flatten_e4, leye_flatten_e4])
    fc_e1 = layers.Dense(128, activation="relu",
                        kernel_regularizer=self._l2,
                        trainable=trainable)(eye_combined)

    # Face layers.
    face_conv_f1 = layers.Conv2D(96, (11, 11), strides=(4, 4),
                                activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(self._face_input)
    face_pool_f1 = layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=(2, 2))(face_conv_f1)
    face_norm_f1 = layers.BatchNormalization(trainable=trainable)(face_pool_f1)

    face_pad_f2 = layers.ZeroPadding2D(padding=(2, 2))(face_norm_f1)
    face_conv_f2 = layers.Conv2D(256, (5, 5), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(face_pad_f2)
    face_pool_f2 = layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=(2, 2))(face_conv_f2)
    face_norm_f2 = layers.BatchNormalization(trainable=trainable)(face_pool_f2)

    face_pad_f3 = layers.ZeroPadding2D(padding=(1, 1))(face_norm_f2)
    face_conv_f3 = layers.Conv2D(384, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(face_pad_f3)

    face_conv_f4 = layers.Conv2D(64, (1, 1), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(face_conv_f3)
    face_flatten_f4 = layers.Flatten()(face_conv_f4)

    face_fc1 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_flatten_f4)
    face_fc2 = layers.Dense(64, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_fc1)

    # Face grid.
    grid_flat = layers.Flatten()(self._grid_input)
    grid_fc1 = layers.Dense(256, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_flat)
    grid_fc2 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_fc1)

    # Concat everything and put through a final FF layer.
    all_concat = layers.Concatenate()([fc_e1, face_fc2, grid_fc2])
    all_fc1 = layers.Dense(128, activation="relu",
                           kernel_regularizer=self._l2,
                           trainable=trainable)(all_concat)
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1)

    return all_fc2

class LargeNetwork(Network):
  """ A variation of MitNetwork with larger convolutions. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Shared eye layers.
    conv_e1 = layers.Conv2D(144, (11, 11), strides=(4, 4), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e1 = layers.BatchNormalization(trainable=trainable)

    pad_e2 = layers.ZeroPadding2D(padding=(2, 2))
    conv_e2 = layers.Conv2D(384, (5, 5), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e2 = layers.BatchNormalization(trainable=trainable)

    pad_e3 = layers.ZeroPadding2D(padding=(1, 1))
    conv_e3 = layers.Conv2D(576, (3, 3), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)

    conv_e4 = layers.Conv2D(64, (1, 1), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    flatten_e4 = layers.Flatten()

    # Left eye stack.
    leye_conv_e1 = conv_e1(self._left_eye_input)
    leye_pool_e1 = pool_e1(leye_conv_e1)
    leye_norm_e1 = norm_e1(leye_pool_e1)

    leye_pad_e2 = pad_e2(leye_norm_e1)
    leye_conv_e2 = conv_e2(leye_pad_e2)
    leye_pool_e2 = pool_e2(leye_conv_e2)
    leye_norm_e2 = norm_e2(leye_pool_e2)

    leye_pad_e3 = pad_e3(leye_norm_e2)
    leye_conv_e3 = conv_e3(leye_pad_e3)

    leye_conv_e4 = conv_e4(leye_conv_e3)
    leye_flatten_e4 = flatten_e4(leye_conv_e4)

    # Right eye stack.
    reye_conv_e1 = conv_e1(self._right_eye_input)
    reye_pool_e1 = pool_e1(reye_conv_e1)
    reye_norm_e1 = norm_e1(reye_pool_e1)

    reye_pad_e2 = pad_e2(reye_norm_e1)
    reye_conv_e2 = conv_e2(reye_pad_e2)
    reye_pool_e2 = pool_e2(reye_conv_e2)
    reye_norm_e2 = norm_e2(reye_pool_e2)

    reye_pad_e3 = pad_e3(reye_norm_e2)
    reye_conv_e3 = conv_e3(reye_pad_e3)

    reye_conv_e4 = conv_e4(reye_conv_e3)
    reye_flatten_e4 = flatten_e4(reye_conv_e4)

    # Concatenate eyes and put through a shared FC layer.
    eye_combined = layers.Concatenate()([reye_flatten_e4, leye_flatten_e4])
    fc_e1 = layers.Dense(128, activation="relu",
                        kernel_regularizer=self._l2,
                        trainable=trainable)(eye_combined)

    # Face layers.
    face_conv_f1 = layers.Conv2D(144, (11, 11), strides=(4, 4),
                                activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(self._face_input)
    face_pool_f1 = layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=(2, 2))(face_conv_f1)
    face_norm_f1 = layers.BatchNormalization(trainable=trainable)(face_pool_f1)

    face_pad_f2 = layers.ZeroPadding2D(padding=(2, 2))(face_norm_f1)
    face_conv_f2 = layers.Conv2D(384, (5, 5), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(face_pad_f2)
    face_pool_f2 = layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=(2, 2))(face_conv_f2)
    face_norm_f2 = layers.BatchNormalization(trainable=trainable)(face_pool_f2)

    face_pad_f3 = layers.ZeroPadding2D(padding=(1, 1))(face_norm_f2)
    face_conv_f3 = layers.Conv2D(576, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(face_pad_f3)

    face_conv_f4 = layers.Conv2D(64, (1, 1), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)(face_conv_f3)
    face_flatten_f4 = layers.Flatten()(face_conv_f4)

    face_fc1 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_flatten_f4)
    face_fc2 = layers.Dense(64, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_fc1)

    # Face grid.
    grid_flat = layers.Flatten()(self._grid_floats)
    grid_fc1 = layers.Dense(256, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_flat)
    grid_fc2 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_fc1)

    # Concat everything and put through a final FF layer.
    all_concat = layers.Concatenate()([fc_e1, face_fc2, grid_fc2])
    all_fc1 = layers.Dense(128, activation="relu",
                          kernel_regularizer=self._l2,
                          trainable=trainable)(all_concat)
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1)

    return all_fc2

class LargeVggNetwork(BwNetwork):
  """ Extension of LargeNetwork that uses a pretrained VGG network to process
  faces. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Get pretrained VGG model for use as a base.
    vgg = keras.applications.vgg19.VGG19(include_top=False,
                                        input_tensor=self._face_floats)
    vgg_out = vgg.outputs

    # Freeze all layers in VGG.
    for layer in vgg.layers:
      layer.trainable = False

    # Shared eye layers.
    conv_e1 = layers.Conv2D(144, (11, 11), strides=(4, 4), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e1 = layers.BatchNormalization(trainable=trainable)

    pad_e2 = layers.ZeroPadding2D(padding=(2, 2))
    conv_e2 = layers.Conv2D(384, (5, 5), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e2 = layers.BatchNormalization(trainable=trainable)

    pad_e3 = layers.ZeroPadding2D(padding=(1, 1))
    conv_e3 = layers.Conv2D(576, (3, 3), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)

    conv_e4 = layers.Conv2D(64, (1, 1), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    flatten_e4 = layers.Flatten()

    # Left eye stack.
    leye_conv_e1 = conv_e1(self._left_eye_input)
    leye_pool_e1 = pool_e1(leye_conv_e1)
    leye_norm_e1 = norm_e1(leye_pool_e1)

    leye_pad_e2 = pad_e2(leye_norm_e1)
    leye_conv_e2 = conv_e2(leye_pad_e2)
    leye_pool_e2 = pool_e2(leye_conv_e2)
    leye_norm_e2 = norm_e2(leye_pool_e2)

    leye_pad_e3 = pad_e3(leye_norm_e2)
    leye_conv_e3 = conv_e3(leye_pad_e3)

    leye_conv_e4 = conv_e4(leye_conv_e3)
    leye_flatten_e4 = flatten_e4(leye_conv_e4)

    # Right eye stack.
    reye_conv_e1 = conv_e1(self._right_eye_input)
    reye_pool_e1 = pool_e1(reye_conv_e1)
    reye_norm_e1 = norm_e1(reye_pool_e1)

    reye_pad_e2 = pad_e2(reye_norm_e1)
    reye_conv_e2 = conv_e2(reye_pad_e2)
    reye_pool_e2 = pool_e2(reye_conv_e2)
    reye_norm_e2 = norm_e2(reye_pool_e2)

    reye_pad_e3 = pad_e3(reye_norm_e2)
    reye_conv_e3 = conv_e3(reye_pad_e3)

    reye_conv_e4 = conv_e4(reye_conv_e3)
    reye_flatten_e4 = flatten_e4(reye_conv_e4)

    # Concatenate eyes and put through a shared FC layer.
    eye_combined = layers.Concatenate()([reye_flatten_e4, leye_flatten_e4])
    fc_e1 = layers.Dense(128, activation="relu",
                        kernel_regularizer=self._l2)(eye_combined)

    # Face layers.
    face_flatten_f4 = layers.Flatten()(vgg_out)

    face_fc1 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_flatten_f4)
    face_fc2 = layers.Dense(64, activation="relu",
                            kernel_regularizer=self._l2)(face_fc1)

    # Face grid.
    grid_flat = layers.Flatten()(self._grid_floats)
    grid_fc1 = layers.Dense(256, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_flat)
    grid_fc2 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_fc1)

    # Concat everything and put through a final FF layer.
    all_concat = layers.Concatenate()([fc_e1, face_fc2, grid_fc2])
    all_fc1 = layers.Dense(128, activation="relu",
                          kernel_regularizer=self._l2)(all_concat)
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1)

    return all_fc2
