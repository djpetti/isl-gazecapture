import keras.backend as K
import keras.layers as layers

from super_layer import SuperLayer


class Residual(SuperLayer):
  """ Implements a residual module as described by He et. al in
  "Deep residual learning for image recognition,"
	Computer Vision and Pattern Recognition, 2016. A layer wrapped with this class
	is evaluated as normal, but the original input is then added back to its
	output. """

  def __init__(self, layers, *args, **kwargs):
    """
    Args:
      layers: List of layers we want to wrap in a residual module.
      projection_strides: The strides to use for the projection layer. Useful if
                          some layer in the model has strides other than 1. """
    self.__projection_strides = kwargs.get("projection_strides", (1, 1))
    if "projection_strides" in kwargs:
      kwargs.pop("projection_strides")

    super(Residual, self).__init__(layers, *args, **kwargs)

  def __check_shape_compatibility(self, inputs):
    """ Checks whether the input and output have compatible shapes.
    Args:
      The inputs to the module.
    Returns:
      The number of input filters, and the number of output filters. """
    input_shape = list(inputs._keras_shape)
    output_shape = list(self.compute_output_shape(input_shape))

    # Assume order is (batch, h, w, filters) initially.
    input_size = input_shape[1:3]
    output_size = output_shape[1:3]
    input_filters = input_shape[3]
    output_filters = output_shape[3]

    if K.image_data_format() == "channels_first":
      # Order is actually (batch, filters, h, w)
      input_size = input_shape[2:]
      output_size = output_shape[2:]
      input_filters = input_shape[1]
      output_filters = output_shape[1]

    proj_size = input_size
    proj_size[0] /= self.__projection_strides[0]
    proj_size[1] /= self.__projection_strides[1]

    # Perform basic sanity checking.
    if (len(input_shape) != 4 or len(output_shape) != 4):
      raise ValueError("Residual() only works with 2D convolution.")
    if proj_size != output_size:
      raise ValueError("Input and output image dims must be compatible.")

    return input_filters, output_filters

  def __call__(self, inputs):
    # Add the wrapped layers.
    raw_outputs = super(Residual, self).__call__(inputs)

    add_back = inputs
    input_filters, output_filters = self.__check_shape_compatibility(inputs)
    if (input_filters != output_filters or self.__projection_strides != (1, 1)):
      # We need an extra 1x1 convolution because the number of input and output
      # filters are not the same, or the image shape is not the same.
      add_back = layers.Conv2D(output_filters, (1, 1),
                               strides=self.__projection_strides,
                               padding="same")(inputs)

    # Perform the addition.
    add = layers.Add()([raw_outputs, add_back])
    # Perform activation.
    return layers.Activation("relu")(add)

class ResNetBlock(Residual):
  """ Performs a block of 2 convolution operations, wrapped in a residual
      module. """

  def __init__(self, filters, kernel_size, **kwargs):
    """ Creates a new block.
    Args:
      filters: Number of filters.
      kernel_size: The size of the kernel.
      expansion_filters: Number of filters to use in the last expansion layer.
                         It defaults to 4 * filters.
      downsample_first: If true, will downsample on the first layer.
      Any additional arguments will be passed
      transparently to the underlying Conv2D layers. """
    # Read expansion_filters argument.
    expansion_filters = kwargs.get("expansion_filters", filters * 4)
    if "expansion_filters" in kwargs:
      kwargs.pop("expansion_filters")
    # Read downsample_first argument.
    downsample = kwargs.get("downsample_first", False)
    if "downsample_first" in kwargs:
      kwargs.pop("downsample_first")

    # For now, it's easier not to mess with unpadded convolutions.
    if "padding" in kwargs:
      raise ValueError("Cannot specify padding for residual block.")
    kwargs["padding"] = "SAME"

    kwargs["activation"] = None
    first_kwargs = kwargs.copy()
    if downsample:
      first_kwargs["strides"] = (2, 2)

    # Initialize the sublayers.
    conv1 = layers.Conv2D(filters, (1, 1), **first_kwargs)
    norm1 = layers.BatchNormalization()
    act1 = layers.Activation("relu")
    conv2 = layers.Conv2D(filters, kernel_size, **kwargs)
    norm2 = layers.BatchNormalization()
    act2 = layers.Activation("relu")
    # We don't want activation for the last layer, since it will be added after
    # the addition operation.
    conv3_kwargs = kwargs.copy()
    conv3_kwargs["activation"] = None
    conv3 = layers.Conv2D(expansion_filters, (1, 1), **conv3_kwargs)
    norm3 = layers.BatchNormalization()

    # Handle downsampling in the residual block.
    proj_strides = (1, 1)
    if downsample:
      proj_strides = (2, 2)

    my_layers = [conv1, norm1, act1, conv2, norm2, act2, conv3, norm3]
    super(ResNetBlock, self).__init__(my_layers,
                                      projection_strides=proj_strides)
