class SuperLayer(object):
  """ Allows the construction of a layer class comprised of other layers. """

  def __init__(self, layers):
    """
    Args:
      layers: A list of the sub-layers that this layer is comprised of. """
    self.__layers = layers

  def __call__(self, inputs):
    """ Actually adds to the graph for the layer.
    Returns:
      The layer output node. """
    # We'll do this by simply calling all the sublayers in sequence.
    for layer in self.__layers:
      inputs = layer(inputs)

    return inputs

  def compute_output_shape(self, input_shape):
    """ Computes the output shape of this layer given the input shape.
    Args:
      input_shape: The shape of the input.
    Returns:
      The shape of the corresponding output. """
    # We'll just do this by calling the same method on our sublayers in
    # sequence.
    for layer in self.__layers:
      input_shape = layer.compute_output_shape(input_shape)

    return input_shape

  def get_weights(self):
    """ Gets the weights for this layer.
    Returns:
      A list of all the weights in the sublayer. """
    all_weights = []
    for layer in self.__layers:
      all_weights.extend(layer.get_weights())

    return all_weights

  def get_sublayers(self):
    """ Gets the list of sublayers.
    Returns:
      The list of sublayers. """
    return self.__layers
