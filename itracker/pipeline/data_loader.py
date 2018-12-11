import collections
import logging
import os
import random
import subprocess

import tensorflow as tf

import preprocess


logger = logging.getLogger(__name__)


def accessible_path(path):
  """ Figures out whether a file exists or not. It looks both locally, and in
  the GCP storage bucket.
  Args:
    path: The path to search for.
  Returns:
    True if the path exists, false otherwise. """
  if path.startswith("gs://"):
    # This is a GCP file.
    try:
      subprocess.check_output(["gsutil", "ls", path])
    except subprocess.CalledProcessError:
      # Does not exist.
      return False
    return True

  else:
    # Local file.
    return os.path.exists(path)

class DataPoint(object):
  """ Structure encapsulating an image and associated metadata. """

  def __init__(self, features):
    """
    Args:
      features: The FeatureSet that we are using. """
    feature_names = features.get_feature_names()

    # Set all the properties.
    for name in feature_names:
      value = features.get_feature_by_name(name)
      setattr(self, name, value)

class FeatureSet(object):
  """ Defines a set of features that we want to load and process from the file.
  """

  def __init__(self, prefix):
    """
    Args:
      prefix: The prefix to use for all feature names. """
    self.__prefix = prefix

    # The feature specifications.
    self.__feature_specs = {}
    # The actual set of feature tensors.
    self.__features = collections.OrderedDict()

    # Set of feature names.
    self.__feature_names = collections.OrderedDict()

  def add_feature(self, name, feature):
    """ Adds a feature to the set.
    Args:
      name: The name of the feature to add.
      feature: The TensorFlow feature. """
    # Add the prefix.
    full_name = "%s/%s" % (self.__prefix, name)
    self.__feature_specs[full_name] = feature

    self.__feature_names[full_name] = name

  def parse_from(self, example):
    """ Parses all the features that were added.
    Args:
      example: The serialized example to parse features from. """
    self.__features = tf.parse_single_example(example, features=self.__feature_specs)

  def get_features(self):
    """
    Returns:
      The full set of features. """
    return self.__features.copy()

  def get_feature_tensors(self):
    """
    Returns:
      A list of the feature tensors. """
    return self.__features.values()

  def get_feature_names(self):
    """
    Returns:
      A set of the names of all the features. """
    return self.__feature_names.values()

  def get_feature_by_name(self, name):
    """ Gets a featue by its name. (Without the prefix.)
    Args:
      name: The name of the feature.
    Returns:
      The feature with that name. """
    # Add the prefix.
    full_name = "%s/%s" % (self.__prefix, name)
    return self.__features[full_name]

  def copy_from(self, new_tensors):
    """ Makes a copy of this feature set with new corresponding feature tensors.
    Args:
      new_tensors: A list of the new feature tensors. It assumes these are in an
                   order corresponding to the order returned by
                   get_feature_tensors.
    Returns:
      The copied feature set. """
    # Make a new set.
    new_set = FeatureSet(self.__prefix)

    # Copy the feature specifications.
    new_set.__feature_specs = self.__feature_specs.copy()
    new_set.__feature_names = self.__feature_names.copy()

    # Add the feature tensors.
    names = self.__features.keys()
    for name, tensor in zip(names, new_tensors):
      new_set.__features[name] = tensor

    return new_set


class DataLoader(object):
  """ Class that is responsible for loading and pre-processing data. """

  # Unique instance number.
  _INSTANCE_NUM = 0

  def __init__(self, records_file, batch_size, image_shape, tpu_flatten=False):
    """
    Args:
      records_file: The TFRecords file to read data from.
      batch_size: The size of batches to read.
      image_shape: The shape of images to load.
      tpu_flatten: Another irritating limitation of the TensorFlow TPU API is
                   that it only supports a single input. If this parameter is
                   True, it will automatically flatten all the inputs to R1
                   tensors, concatenate them together, and produce that as a
                   single input. Note that the inputs must be manually
                   un-flattened by the model training code. """
    if not accessible_path(records_file):
      # If we don't check this, TensorFlow gives us a really confusing and
      # hard-to-debug error later on.
      raise ValueError("File '%s' does not exist." % (records_file))
    if len(image_shape) != 3:
      raise ValueError("Image shape must be of length 3.")

    self._image_shape = image_shape
    self._records_file = records_file
    self._batch_size = batch_size
    self.__tpu_flatten = tpu_flatten

    # Set unique instance number.
    self.__id = DataLoader._INSTANCE_NUM
    DataLoader._INSTANCE_NUM += 1

    # Create a default preprocessing pipeline.
    self.__pipeline = preprocess.Pipeline()

  def __decode_and_preprocess(self, feature_tensors):
    """ Target for map_fn that decodes and preprocesses individual images.
    Args:
      feature_tensors: Individual feature tensors passed in by map_fn.
    Returns:
      A list of the preprocessed image nodes. """
    # Create a new feature map with the individual feature tensors.
    single_features = self._features.copy_from(feature_tensors)

    # Find the encoded image feature.
    jpeg = single_features.get_feature_by_name("image")

    # Decode the image.
    image = tf.image.decode_jpeg(jpeg[0])
    # Resize the image to a defined shape.
    image = tf.reshape(image, self._image_shape)

    # Create a data point object.
    data_point = DataPoint(single_features)
    # Use the decoded image instead of the encoded one.
    data_point.image = image

    # Pre-process the image.
    return self._build_preprocessing_stage(data_point)

  def __preprocess_one(self, serialized):
    """ Decodes and preprocesses and single serialized example.
    Args:
      serialized: The serialized example.
    Returns:
      The preprocessed tensors. """
    # Deserialize the example.
    self._features.parse_from(serialized)

    # Decode and pre-process in parallel.
    feature_tensors = self._features.get_feature_tensors()
    images = self.__decode_and_preprocess(feature_tensors)

    # Create the batches.
    labels = self._features.get_feature_by_name("dots")
    data = None
    if self.__tpu_flatten:
      # Produce a single output, flattened for the TPU.
      data = self.__flatten_for_tpu(images)
    else:
      # Produce a dictionary mapping input names to tensors.
      data = self.__associate_with_inputs(images)

    return data, labels

  def __build_loader_stage(self):
    """ Builds the pipeline stages that actually loads data from the disk. """
    # 256 MB buffer for reading from the dataset.
    buffer_size = 256 * 1024 * 1024
    # Define a dataset.
    common_reader = tf.data.TFRecordDataset(self._records_file,
                                            buffer_size=buffer_size) \
                                    .shuffle(int(self._batch_size * 1.5))

    # Initially, we want to start at a random point.
    start_at = random.randint(0, self._batch_size * 100)
    logger.debug("Starting at example %s." % (start_at))
    first_reader = common_reader.skip(start_at)
    reader = first_reader.concatenate(common_reader.repeat())
    # Fused mapping and batching operation for performance.
    reader = reader.apply(tf.data.experimental.map_and_batch( \
        map_func=self.__preprocess_one, batch_size=self._batch_size,
        num_parallel_batches=2,
        drop_remainder=True))

    # Prefetch data. Since we've already batched, the argument is the number of
    # batches.
    reader = reader.prefetch(tf.contrib.data.AUTOTUNE)
    logger.info("Dataset: %s" % (str(reader)))

    self.__dataset = reader

  def __flatten_for_tpu(self, out_nodes):
    """ Flattens map output nodes into a single R1 output for the TPU.
    Args:
      out_nodes: The output nodes from the map call.
    Returns:
      A single R1 tensor containing all the nodes. """
    flat_list = []
    for node in out_nodes:
      flat_node = tf.reshape(node, [-1])
      flat_list.append(flat_node)

    # Concatenate them all end-to-end.
    return tf.concat(flat_list, 0)

  def __associate_with_inputs(self, out_nodes):
    """ Associates map output nodes with their respective named inputs.
    Args:
      out_nodes: The output nodes from the map call.
    Returns:
      A dictionary mapping pipelines to nodes. """
    pipelines = self.__pipeline.get_leaf_pipelines()

    mapping = {}
    for pipeline, node in zip(pipelines, out_nodes):
      # Get the associated input name.
      input_name = pipeline.get_associated_input_name()
      if input_name is None:
        raise ValueError("All leaf pipelines must be associated with inputs.")

      mapping[input_name] = node

    return mapping

  def _init_feature_set(self, prefix):
    """ Initializes the FeatureSet to use for this loader. This must be
    overriden by a subclass.
    Args:
      prefix: The prefix to use for feature names.
    Returns:
      The initialized FeatureSet. """
    raise NotImplementedError( \
        "_init_feature_set() must be implemented by subclass.")

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
    # Initialize the feature set.
    self._features = self._init_feature_set(prefix)
    # Build the loader stage.
    self.__build_loader_stage()

  def get_data(self):
    """
    Returns:
      The loaded data, as a Dataset. """
    return self.__dataset

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
