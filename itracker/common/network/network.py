import logging

import tensorflow as tf

from ...pipeline import keras_utils


logger = logging.getLogger(__name__)


optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers
layers = tf.keras.layers


class Network(object):
  """ Represents a network. """

  def __init__(self, input_shape, eye_shape=None, fine_tune=False,
               data_tensors=None, eye_preproc=None, face_preproc=None,
               l2_reg=0.0005, flat_inputs=False, **kwargs):
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
                    preprocessed.
      l2_reg: The alpha value to use for l2 regularization.
      flat_inputs: Whether we receive the inputs as a flattened tensor, and have
                   to reshape them manually. This is how inputs are received on
                   the TPU. """
    self.__data_tensors = data_tensors
    self.__eye_preproc = eye_preproc
    self.__face_preproc = face_preproc
    self.__reg_alpha = l2_reg
    self.__flat_inputs = flat_inputs
    self._fine_tune = fine_tune
    self._input_shape = input_shape

    # The eventual outputs of the model.
    self._outputs = None

    self._eye_shape = self._input_shape
    if eye_shape is not None:
      self._eye_shape = eye_shape

  def __unflatten_inputs(self):
    """ Converts the inputs back from their flattened state into a more normal
    one.
    Returns:
      Tensors for the left eye, right eye, face and grid. """
    def unflatten_layer(flat_inputs):
      """ Lambda layer that takes the flat inputs and unflattens them.
      Args:
        flat_inputs: The flattened input tensor.
      Returns:
        The unflattened inputs. """
      # Extract the individual inputs.
      left_eye = flat_inputs[:, 0:eye_nodes]
      right_eye = flat_inputs[:, eye_nodes:2 * eye_nodes]
      face = flat_inputs[:, 2 * eye_nodes:2 * eye_nodes + face_nodes]
      grid = flat_inputs[:, input_nodes - grid_nodes:input_nodes]

      # Reshape them.
      left_eye = tf.reshape(left_eye, (-1,) + self._eye_shape)
      right_eye = tf.reshape(right_eye, (-1,) + self._eye_shape)
      face = tf.reshape(face, (-1,) + self._input_shape)
      grid = tf.reshape(grid, (-1, 25, 25))

      #return left_eye, right_eye, face, grid
      return left_eye, right_eye, face, grid


    # Compute the flattened input shape.
    face_nodes = 1
    for dim in self._input_shape:
      face_nodes *= dim
    eye_nodes = 1
    for dim in self._eye_shape:
      eye_nodes *= dim
    grid_nodes = (25 * 25)
    input_nodes = face_nodes + eye_nodes * 2 + grid_nodes
    logger.debug("Expecting %d input nodes." % (input_nodes))

    # Create a single input for the flattened data.
    self._flat_input = tf.keras.Input(shape=(input_nodes,), name="flat_input")

    # Lambda layer to unflatten.
    return layers.Lambda(unflatten_layer)(self._flat_input)

  def __create_inputs(self):
    """ Creates all the network inputs.
    Returns:
      Tensors for the left eye, right eye, face and grid. """
    def input_creator(tensor):
      """ Chooses the input layer creator function to use.
      Args:
        tensor: Optional input tensor to use. """
      if tensor is not None:
        # We want to use the pipeline input creator.
        return keras_utils.pipeline_input
      else:
        # Use the normal creator.
        return tf.keras.Input

    leye = None
    reye = None
    face = None
    grid = None
    if self.__data_tensors:
      leye, reye, face, grid = self.__data_tensors

    # Create inputs.
    input_class = keras_utils.pipeline_input
    left_eye_input = input_creator(leye)(shape=self._eye_shape,
                                         tensor=leye,
                                         name="left_eye_input")
    right_eye_input = input_creator(reye)(shape=self._eye_shape,
                                          tensor=reye,
                                          name="right_eye_input")
    face_input = input_creator(face)(shape=self._input_shape,
                                     tensor=face,
                                     name="face_input")
    grid_input = input_creator(grid)(shape=(25, 25), tensor=grid,
                                     name="grid_input")

    return left_eye_input, right_eye_input, face_input, grid_input

  def _apply_all(self, inputs, layers):
    """ Creates a stack of layers, where each layer is applied to the output of
    the previous one.
    Args:
      inputs: The initial inputs to apply the stack on.
      layers: The layers to apply.
    Returns:
      The output from the last layer. """
    next_input = inputs

    for layer in layers:
      next_input = layer(next_input)

    return next_input

  def _build_common(self):
    """ Build the network components that are common to all. """
    if self.__flat_inputs:
      logger.info("Using flattened input.")
      # Use the single flat input.
      self._left_eye_input, self._right_eye_input, self._face_input, \
          self._grid_input = self.__unflatten_inputs()
    else:
      logger.info("Using standard inputs.")
      # Use the standard four-input design.
      self._left_eye_input, self._right_eye_input, self._face_input, \
          self._grid_input = self.__create_inputs()

    # Add preprocessing layers.
    self._left_eye_node = self._left_eye_input
    self._right_eye_node = self._right_eye_input
    if self.__eye_preproc is not None:
      self._left_eye_node = self.__eye_preproc(self._left_eye_input)
      self._right_eye_node = self.__eye_preproc(self._right_eye_input)

    self._face_node = self._face_input
    if self.__face_preproc is not None:
      self._face_node = self.__face_preproc(self._face_input)

    # L2 regularizer for weight decay.
    logger.debug("Using regularization: %f" % (self.__reg_alpha))
    self._l2 = regularizers.l2(self.__reg_alpha)

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
    inputs = [self._left_eye_input, self._right_eye_input, self._face_input,
              self._grid_input]
    if self.__flat_inputs:
      # Use flattened input instead.
      inputs=self._flat_input
    model = tf.keras.Model(inputs=inputs,
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
      logger.debug("Building tf.keras.")

      # Build the common parts.
      self._build_common()
      # Build the custom parts.
      self._outputs = self._build_custom()

    # Create the model.
    logger.debug("Creating new model: %s", self.__class__.__name__)
    return self._create_model()

  def prepare_labels(self, labels):
    """ Sometimes the network architecture requires that we modify the labels in
    some way before using them. This method provides a convenient way to do it.
    By default, it is just an identity operation.
    Args:
      labels: The labels to prepare.
    Returns:
      The modified labels. """
    return labels
