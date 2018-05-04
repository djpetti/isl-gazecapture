import tensorflow as tf


class DataPoint(object):
  """ Structure encapsulating an image and associated metadata. """

  def __init__(self, *args, **kwargs):
    # The actual image data.
    self.image = kwargs.get("image")

    # The dot location.
    self.dot = kwargs.get("dot")
    # The face size.
    self.face_size = kwargs.get("face_size")
    # The left eye bounding box.
    self.leye_box = kwargs.get("leye_box")
    # The right eye bounding box.
    self.reye_box = kwargs.get("reye_box")
    # The face grid bounding box.
    self.grid_box = kwargs.get("grid_box")


class DataLoader(object):
  """ Class that is responsible for loading and pre-processing data. """

  def __init__(self, image_shape):
    """
    Args:
      image_shape: The shape of images to load. """
    self._image_shape = image_shape

  def __decode_and_preprocess(self, features):
    # Unpack the reatures sequence.
    jpeg, dot, face_size, leye_box, reye_box, grid_box = features

    # Decode the image.
    image = tf.image.decode_jpeg(jpeg[0])
    # Resize the image to a defined shape.
    image = tf.reshape(image, self._image_shape)

    # Create a data point object.
    data_point = DataPoint(image=image, dot=dot, face_size=face_size,
                           leye_box=leye_box, reye_box=reye_box,
                           grid_box=grid_box)

    # Pre-process the image.
    return self._build_preprocessing_stage(data_point)

  def __build_loader_stage(self, records_file, prefix, batch_size):
    """ Builds the pipeline stages that actually loads data from the disk.
    Args:
      records_file: The TFRecords file to load data from.
      prefix: The prefix for the feature names to load.
      batch_size: The size of each batch to load.
    Returns:
      The features that it loaded from the file, as a sequence of tensors. """
    feature = {"%s/dots" % (prefix): tf.FixedLenFeature([2], tf.float32),
               "%s/face_size" % (prefix): tf.FixedLenFeature([2], tf.float32),
               "%s/leye_box" % (prefix): tf.FixedLenFeature([4], tf.float32),
               "%s/reye_box" % (prefix): tf.FixedLenFeature([4], tf.float32),
               "%s/grid_box" % (prefix): tf.FixedLenFeature([4], tf.float32),
               "%s/image" % (prefix): tf.FixedLenFeature([1], tf.string)}

    # Create queue for filenames, which is a little silly since we only have one
    # file.
    filename_queue = tf.train.string_input_producer([records_file])

    # Define a reader and read the next record.
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    # Prepare random batches.
    batch = tf.train.shuffle_batch([serialized_examples],
                                   batch_size=batch_size,
                                   capacity=batch_size * 10,
                                   min_after_dequeue=batch_size / 3,
                                   num_threads=8)

    # Deserialize the example.
    features = tf.parse_example(batch, features=feature)

    # Convert data into a tensor sequence.
    image = features["%s/image" % (prefix)]
    dot = features["%s/dots" % (prefix)]
    face_size = features["%s/face_size" % (prefix)]
    leye_box = features["%s/leye_box" % (prefix)]
    reye_box = features["%s/reye_box" % (prefix)]
    grid_box = features["%s/grid_box" % (prefix)]

    return (image, dot, face_size, leye_box, reye_box, grid_box)

  def _build_preprocessing_stage(self, data_point):
    """ Performs preprocessing on an image node.
    Args:
      data_point: The DataPoint object to use for preprocessing.
    Returns:
      The preprocessed image node. """
    # TODO (danielp): Preprocessing.
    return data_point.image

  def _build_pipeline(self, records_file, prefix, batch_size):
    """ Builds the entire pipeline for loading and preprocessing data.
    Args:
      records_file: The TFRecords file to load data from.
      prefix: The prefix that is used for the feature names.
      batch_size: The batch size to use. """
    # Build the loader stage.
    features = self.__build_loader_stage(records_file, prefix, batch_size)

    # Decode and pre-process in parallel.
    images = tf.map_fn(self.__decode_and_preprocess, features, dtype=tf.uint8,
                       back_prop=False, parallel_iterations=8)

    # Create the batches.
    dots = features[1]
    self.__x = images
    self.__y = dots

  def get_data(self):
    """
    Returns:
      The node for the loaded data. """
    return self.__x

  def get_labels(self):
    """
    Returns:
      The node for the loaded labels. """
    return self.__y

class TrainDataLoader(DataLoader):
  """ DataLoader for training data. """

  def __init__(self, records_file, batch_size, image_shape):
    """
    Args:
      records_file: The TFRecords file to load data from.
      batch_size: The size of each batch to load.
      image_shape: The shape of images to load. """
    super(TrainDataLoader, self).__init__(image_shape)

    self._build_pipeline(records_file, "train", batch_size)

class TestDataLoader(DataLoader):
  """ DataLoader for testing data. """

  def __init__(self, records_file, batch_size, image_shape):
    """
    Args:
      records_file: The TFRecords file to load data from.
      batch_size: The size of each batch to load.
      image_shape: The shape of images to load. """
    super(TestDataLoader, self).__init__(image_shape)

    self._build_pipeline(records_file, "test", batch_size)
