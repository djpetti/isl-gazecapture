import tensorflow as tf

import preprocess


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

  def __init__(self, records_file, batch_size, image_shape):
    """
    Args:
      records_file: The TFRecords file to read data from.
      batch_size: The size of batches to read.
      image_shape: The shape of images to load. """
    self._image_shape = image_shape
    self._records_file = records_file
    self._batch_size = batch_size

    # Create a default preprocessing pipeline.
    self.__pipeline = preprocess.Pipeline()

  def __decode_and_preprocess(self, features):
    """ Target for map_fn that decodes and preprocesses individual images.
    Args:
      features: The input features that were loaded.
    Returns:
      A list of the preprocessed image nodes. """
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

  def __build_loader_stage(self, prefix):
    """ Builds the pipeline stages that actually loads data from the disk.
    Args:
      prefix: The prefix for the feature names to load.
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
    filename_queue = tf.train.string_input_producer([self._records_file])

    # Define a reader and read the next record.
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    # Prepare random batches.
    batch = tf.train.shuffle_batch([serialized_examples],
                                   batch_size=self._batch_size,
                                   capacity=self._batch_size * 30,
                                   min_after_dequeue=self._batch_size / 3,
                                   num_threads=16)

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

  def __associate_with_pipelines(self, out_nodes):
    """ Associates map_fn output nodes with their respective pipelines.
    Args:
      out_nodes: The output nodes from the map_fn call.
    Returns:
      A dictionary mapping pipelines to nodes. """
    pipelines = self.__pipeline.get_leaf_pipelines()

    mapping = {}
    for pipeline, node in zip(pipelines, out_nodes):
      mapping[pipeline] = node

    return mapping

  def _build_preprocessing_stage(self, data_point):
    """ Performs preprocessing on an image node.
    Args:
      data_point: The DataPoint object to use for preprocessing.
    Returns:
      The preprocessed image nodes. """
    # Convert the images to floats before preprocessing.
    data_point.image = tf.cast(data_point.image, tf.float32)

    # Build the entire pipeline.
    self.__pipeline.build(data_point)
    data_points = self.__pipeline.get_outputs()

    # Extract the image nodes.
    image_nodes = []
    for data_point in data_points:
      image_nodes.append(data_point.image)

    return image_nodes

  def _build_pipeline(self, prefix):
    """ Builds the entire pipeline for loading and preprocessing data.
    Args:
      prefix: The prefix that is used for the feature names. """
    # Build the loader stage.
    features = self.__build_loader_stage(prefix)

    # Tensorflow expects us to tell it the shape of the output beforehand, so we
    # need to compute that.
    dtype = [tf.float32] * self.__pipeline.get_num_outputs()
    # Decode and pre-process in parallel.
    images = tf.map_fn(self.__decode_and_preprocess, features, dtype=dtype,
                       back_prop=False, parallel_iterations=16)

    # Create the batches.
    dots = features[1]
    self.__x = self.__associate_with_pipelines(images)
    self.__y = dots

  def get_data(self):
    """
    Returns:
      The loaded data, as a dict indexed by pipelines. """
    return self.__x

  def get_labels(self):
    """
    Returns:
      The node for the loaded labels. """
    return self.__y

  def get_pipeline(self):
    """ Gets the preprocessing pipeline object so that preprocessing stages can
    be added.
    Returns:
      The preprocessing pipeline. Add stages to this pipeline to control the
      preprocessing step. """
    return self.__pipeline

  def build(self):
    """ Builds the graph. This must be called before using the loader. """
    raise NotImplementedError("Must be implemented by subclass.")


class TrainDataLoader(DataLoader):
  """ DataLoader for training data. """

  def build(self):
    self._build_pipeline("train")

class TestDataLoader(DataLoader):
  """ DataLoader for testing data. """

  def build(self):
    self._build_pipeline("test")

class ValidDataLoader(DataLoader):
  """ DataLoader for validation data. """

  def build(self):
    self._build_pipeline("val")
