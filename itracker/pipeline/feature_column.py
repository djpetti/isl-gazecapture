import tensorflow as tf


class _FeatureColumnWrapper(object):
  """ TensorFlow doesn't really like to give us access to the underlying
  feature_column classes, so the best way to add functionality seems to be to
  wrap existing feature_columns and extend them through compositing. That's what
  this class does. """

  # Invisible key that is passed to the constructor to indicate that it is being
  # called internally.
  __create_key = object()

  @classmethod
  def wrap(cls, column, *args, **kwargs):
    """ Wraps a new feature column.
    Args:
      column: The feature column to wrap.
      All other arguments are transparently passed to the constructor.
    Returns:
      The wrapped version of column, or the same thing if the column was
      already wrapped. """
    if isinstance(column, _FeatureColumnWrapper):
      # It's already wrapped.
      return column

    # Otherwise, make a new one.
    return cls(cls.__create_key, column, *args, **kwargs)

  @classmethod
  def make_parse_example_spec(cls, columns):
    """ Correctly produces a parsing spec dictionary from all wrapped columns.
    Args:
      columns: A list of wrapped columns to produce the spec from.
    Returns:
      The created spec dictionary. """
    # Create and merge specs for all columns.
    spec = {}
    for column in columns:
      spec.update(column._make_spec())

    return spec

  def __init__(self, _key, column):
    """ Creates a new wrapped feature column.
    NOTE: This is not meant to be called publicly.
    Args:
      _key: Used internally. Ignore.
      column: The column to wrap. """
    # Check that the factory was used.
    if _key != _FeatureColumnWrapper.__create_key:
      raise ValueError("Use wrap() instead of instantiating directly.")

    self.__column = column
    self.__prefix = None

  def _make_spec(self):
    """ Generates a spec dictionary with a single entry for only this column.
    Returns:
      The generated spec dictionary. """
    raw_spec = tf.feature_column.make_parse_example_spec([self.__column])

    # Convert the key to include the prefix.
    spec = {}
    spec[self.get_name_with_prefix()] = raw_spec[self.__column.key]
    return spec

  def inject_prefix(self, prefix):
    """ Allows the loader to set the prefix after the column has been
    instantiated.
    Args:
      prefix: The prefix to set. """
    self.__prefix = prefix

  def get_name(self):
    """
    Returns:
      The name of the feature column. (Without the prefix.) """
    return self.__column.key

  def get_name_with_prefix(self):
    """
    Returns:
      The name of the feature column with the prefix prepended. """
    return "%s/%s" % (self.__prefix, self.__column.key)

  def get_feature_tensor(self, parsed_features):
    """ Finds the value in a set of parsed features that is associated with this
    column.
    Args:
      parsed_features: The parsed features, generally the output from
      tf.parse_example() or a related function.
    Returns:
      The associated value. """
    # Find the associated tensor.
    name = self.get_name_with_prefix()
    return parsed_features[name]

class _ImageColumnWrapper(_FeatureColumnWrapper):
  """ Wraps a feature column that represents a compressed image. """

  def __init__(self, _key, column, image_shape):
    """
    Args:
      _key: Used internally. Ignore.
      column: The column to wrap.
      image_shape: The shape of the extracted image. """
    super(_ImageColumnWrapper, self).__init__(_key, column)

    self.__image_shape = image_shape

  def __decode_image(self, compressed):
    """ Function that decodes a single image.
    Args:
      compressed: The compressed image.
    Returns:
      The decoded image, as float32s. """
    # Decompress the image.
    image = tf.image.decode_jpeg(compressed[0])
    # Resize the image to a defined shape.
    image = tf.reshape(image, self.__image_shape)
    # Convert to floats.
    return tf.cast(image, tf.float32)

  def _make_spec(self):
    # Since there's a transformation here between our output and our input,
    # make_parse_example_spec isn't going to get it right, so we need to
    # generate the spec manually.
    key = self.get_name_with_prefix()
    spec = {key: tf.FixedLenFeature([1], tf.string)}
    return spec

  def get_feature_tensor(self, parsed_features):
    """ Same as superclass, but returns the decompressed image, not the
    compressed version. """
    # Find the associated tensor.
    compressed = super(_ImageColumnWrapper, self) \
        .get_feature_tensor(parsed_features)

    # Decode all the images in the batch.
    images = tf.map_fn(self.__decode_image, compressed, dtype=tf.float32,
                       back_prop=False, parallel_iterations=16)

    return images

def wrap(*args, **kwargs):
  """ Convenience method for wrapping existing feature columns. It forwards
  seamlessly to the corresponding classmethod. """
  return _FeatureColumnWrapper.wrap(*args, **kwargs)

def make_parse_example_spec(*args, **kwargs):
  """ Convenience method for making a parsing spec dictionary. It forwards
  seamlessly to the corresponding classmethod. """
  return _FeatureColumnWrapper.make_parse_example_spec(*args, **kwargs)

def image_column(key, shape):
  """ Creates a column that represents an image that is read from the database
  in compressed form.
  Args:
    key: String key to use for the column.
    shape: The shape of the image in the column. """
  # Create a new image column.
  column = tf.feature_column.numeric_column(key, shape=shape, dtype=tf.float32)
  return _ImageColumnWrapper.wrap(column, shape)
