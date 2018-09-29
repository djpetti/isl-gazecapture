import logging

from keras.models import Model, load_model
import keras.applications as applications
import keras.backend as K
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers

import tensorflow as tf

from ..pipeline import keras_utils


logger = logging.getLogger(__name__)


class Network(object):
  """ Represents a network. """

  def __init__(self, input_shape, eye_shape=None, fine_tune=False,
               data_tensors=None, eye_preproc=None, face_preproc=None):
    """ Creates a new network.
    Args:
      input_shape: The input shape to the network.
      eye_shape: Specify the shape of the eye inputs, if it is different from
                 face input shape.
      fine_tune: Whether we are fine-tuning the model.
      data_tensors: If specified, the set of output tensors from the pipeline,
                    which will be used to build the model.
      eye_preproc: Optional custom layer to use for preprocessing the eye data.
                   If not present, it assumes that the input arrives
                   preprocessed.
      face_preproc: Optional custom layer to use for preprocessing the face
                    data. If not present, it assumes that the input arrives
                    preprocessed. """
    self.__data_tensors = data_tensors
    self.__eye_preproc = eye_preproc
    self.__face_preproc = face_preproc
    self._fine_tune = fine_tune
    self._input_shape = input_shape

    # The eventual outputs of the model.
    self._outputs = None

    self._eye_shape = self._input_shape
    if eye_shape is not None:
      self._eye_shape = eye_shape

  def _build_common(self):
    """ Build the network components that are common to all. """
    def input_creator(tensor):
      """ Chooses the input layer creator function to use.
      Args:
        tensor: Optional input tensor to use. """
      if tensor is not None:
        # We want to use the pipeline input creator.
        return keras_utils.pipeline_input
      else:
        # Use the normal creator.
        return layers.Input

    # L2 regularizer for weight decay.
    self._l2 = regularizers.l2(0.0005)

    leye = None
    reye = None
    face = None
    grid = None
    if self.__data_tensors:
      leye, reye, face, grid = self.__data_tensors

    # Create inputs.
    input_class = keras_utils.pipeline_input
    self._left_eye_input = input_creator(leye)(shape=self._eye_shape,
                                               tensor=leye,
                                               name="left_eye_input")
    self._right_eye_input = input_creator(reye)(shape=self._eye_shape,
                                                tensor=reye,
                                                name="right_eye_input")
    self._face_input = input_creator(face)(shape=self._input_shape,
                                           tensor=face,
                                           name="face_input")
    self._grid_input = input_creator(grid)(shape=(25, 25), tensor=grid,
                                           name="grid_input")

    # Add preprocessing layers.
    self._left_eye_node = self._left_eye_input
    self._right_eye_node = self._right_eye_input
    if self.__eye_preproc is not None:
      self._left_eye_node = self.__eye_preproc(self._left_eye_input)
      self._right_eye_node = self.__eye_preproc(self._right_eye_input)

    self._face_node = self._face_input
    if self.__face_preproc is not None:
      self._face_node = self.__face_preproc(self._face_input)

  def _build_custom(self):
    """ Builds the custom part of the network. Override this in a subclass.
    Returns:
      The outputs that will be used in the model. """
    raise NotImplementedError("Must be implemented by subclass.")

  def _create_model(self):
    """ Creates the model. When this is called, we can safely assume that all
    the inputs and outputs are initialized.
    Returns:
      The model that it created. """
    model = Model(inputs=[self._left_eye_input, self._right_eye_input,
                          self._face_input, self._grid_input],
                  outputs=self._outputs)
    model.summary()

    return model

  def build(self):
    """ Builds the network.
    Note that the implementation creates the layers once and then recycles them
    for multiple calls to build().
    Returns:
      The built model. """
    # Only create new layers if we don't have old ones already.
    if self._outputs is None:
      logger.debug("Building layers.")

      # Build the common parts.
      self._build_common()
      # Build the custom parts.
      self._outputs = self._build_custom()

    # Create the model.
    logger.debug("Creating new model: %s", self.__class__.__name__)
    return self._create_model()

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
    leye_conv_e1 = conv_e1(self._left_eye_node)
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
    reye_conv_e1 = conv_e1(self._right_eye_node)
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
                                trainable=trainable)(self._face_node)
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
    leye_conv_e1 = conv_e1(self._left_eye_node)
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
    reye_conv_e1 = conv_e1(self._right_eye_node)
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
                                trainable=trainable)(self._face_node)
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

class LargeVggNetwork(Network):
  """ Extension of LargeNetwork that uses a pretrained VGG network to process
  faces. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Get pretrained VGG model for use as a base.
    vgg = applications.vgg19.VGG19(include_top=False,
                                   input_shape=self._input_shape)
    vgg_out = vgg(self._face_node)

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
    leye_conv_e1 = conv_e1(self._left_eye_node)
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
    reye_conv_e1 = conv_e1(self._right_eye_node)
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
    eye_drop = layers.Dropout(0.5)(eye_combined)
    fc_e1 = layers.Dense(128, activation="relu",
                        kernel_regularizer=self._l2)(eye_drop)

    # Face layers.
    face_flatten_f4 = layers.Flatten()(vgg_out)

    face_drop = layers.Dropout(0.5)(face_flatten_f4)
    face_fc1 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_drop)
    face_fc2 = layers.Dense(64, activation="relu",
                            kernel_regularizer=self._l2)(face_fc1)

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
                          kernel_regularizer=self._l2)(all_concat)
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1)

    return all_fc2

class ResidualNetwork(Network):
  """ Uses residual-module-based architecture for eye feature extraction. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Get pretrained VGG model for use as a base.
    vgg = applications.vgg19.VGG19(include_top=False,
                                   input_shape=self._input_shape)
    vgg_out = vgg(self._face_node)

    # Freeze all layers in VGG.
    for layer in vgg.layers:
      layer.trainable = False

    # Shared eye layers.
    conv1_e1 = layers.Conv2D(96, (11, 11), strides=(4, 4), activation="relu",
                             kernel_regularizer=self._l2)
    conv1_e2 = layers.Conv2D(96, (1, 1), activation="relu",
                             kernel_regularizer=self._l2)
    norm1 = layers.BatchNormalization(trainable=trainable)

    pool1 = layers.MaxPooling2D(pool_size=3, strides=(2, 2))

    pad2 = layers.ZeroPadding2D(padding=(2, 2))
    conv2_e1 = layers.Conv2D(171, (3, 3), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv2_e2 = layers.Conv2D(171, (3, 3), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv2_e3 = layers.Conv2D(171, (1, 1), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)

    shortcut2 = layers.Conv2D(171, (1, 1), activation="relu",
                              kernel_regularizer=self._l2, trainable=trainable)
    norm2 = layers.BatchNormalization(trainable=trainable)

    pool2 = layers.MaxPooling2D(pool_size=3, strides=(2, 2))

    pad3 = layers.ZeroPadding2D(padding=(1, 1))
    conv3_e1 = layers.Conv2D(256, (3, 3), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv3_e2 = layers.Conv2D(256, (1, 1), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv3_e3 = layers.Conv2D(256, (1, 1), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)

    shortcut3 = layers.Conv2D(256, (1, 1), activation="relu",
                              kernel_regularizer=self._l2, trainable=trainable)
    norm3 = layers.BatchNormalization(trainable=trainable)

    conv4_e1 = layers.Conv2D(64, (1, 1), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    flatten4 = layers.Flatten()

    # Left eye stack.
    leye_conv1_e1 = conv1_e1(self._left_eye_node)
    leye_conv1_e2 = conv1_e2(leye_conv1_e1)
    #leye_norm1 = norm1(leye_conv1_e2)
    leye_pool1 = pool1(leye_conv1_e2)

    leye_pad2 = pad2(leye_pool1)
    leye_conv2_e1 = conv2_e1(leye_pad2)
    leye_conv2_e2 = conv2_e2(leye_conv2_e1)
    leye_conv2_e3 = conv2_e3(leye_conv2_e2)
    leye_shortcut2 = shortcut2(leye_pool1)
    leye_mod2 = layers.Add()([leye_conv2_e3, leye_shortcut2])
    #leye_norm2 = norm2(leye_mod2)
    leye_pool2 = pool2(leye_mod2)

    leye_pad3 = pad3(leye_pool2)
    leye_conv3_e1 = conv3_e1(leye_pad3)
    leye_conv3_e2 = conv3_e2(leye_conv3_e1)
    leye_conv3_e3 = conv3_e3(leye_conv3_e2)
    leye_shortcut3 = shortcut3(leye_pool2)
    leye_mod3 = layers.Add()([leye_conv3_e3, leye_shortcut3])
    #leye_norm3 = norm3(leye_mod3)

    leye_conv4_e1 = conv4_e1(leye_mod3)
    leye_flatten4 = flatten4(leye_conv4_e1)

    # Right eye stack.
    reye_conv1_e1 = conv1_e1(self._left_eye_node)
    reye_conv1_e2 = conv1_e2(reye_conv1_e1)
    #reye_norm1 = norm1(reye_conv1_e2)
    reye_pool1 = pool1(reye_conv1_e2)

    reye_pad2 = pad2(reye_pool1)
    reye_conv2_e1 = conv2_e1(reye_pad2)
    reye_conv2_e2 = conv2_e2(reye_conv2_e1)
    reye_conv2_e3 = conv2_e3(reye_conv2_e2)
    reye_shortcut2 = shortcut2(reye_pool1)
    reye_mod2 = layers.Add()([reye_conv2_e3, reye_shortcut2])
    #reye_norm2 = norm2(reye_mod2)
    reye_pool2 = pool2(reye_mod2)

    reye_pad3 = pad3(reye_pool2)
    reye_conv3_e1 = conv3_e1(reye_pad3)
    reye_conv3_e2 = conv3_e2(reye_conv3_e1)
    reye_conv3_e3 = conv3_e3(reye_conv3_e2)
    reye_shortcut3 = shortcut3(reye_pool2)
    reye_mod3 = layers.Add()([reye_conv3_e3, reye_shortcut3])
    #reye_norm3 = norm3(reye_mod3)

    reye_conv4_e1 = conv4_e1(reye_mod3)
    reye_flatten4 = flatten4(reye_conv4_e1)

    # Concatenate eyes and put through a shared FC layer.
    eye_combined = layers.Concatenate()([reye_flatten4, leye_flatten4])
    eye_drop = layers.Dropout(0.5)(eye_combined)
    fc_e1 = layers.Dense(128, activation="relu",
                        kernel_regularizer=self._l2)(eye_drop)

    # Face layers.
    face_flatten_f4 = layers.Flatten()(vgg_out)

    face_drop = layers.Dropout(0.5)(face_flatten_f4)
    face_fc1 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_drop)
    face_fc2 = layers.Dense(64, activation="relu",
                            kernel_regularizer=self._l2)(face_fc1)

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
                          kernel_regularizer=self._l2)(all_concat)
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1)

    return all_fc2
