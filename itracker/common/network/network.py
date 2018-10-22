import logging

from keras.models import Model
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers

from ...pipeline import keras_utils


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

  def prepare_labels(self, labels):
    """ Sometimes the network architecture requires that we modify the labels in
    some way before using them. This method provides a convenient way to do it.
    By default, it is just an identity operation.
    Args:
      labels: The labels to prepare.
    Returns:
      The modified labels. """
    return labels
