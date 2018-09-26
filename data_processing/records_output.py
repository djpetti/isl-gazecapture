import tensorflow as tf


class Saver(object):
  """ Handles writing the output TFRecords file. """

  def __init__(self, randomizer, split, writer):
    """
    Args:
      randomizer: The FrameRandomizer to draw examples from.
      split: The name of this split.
      writer: The TFRecords writer to write the data with. """
    self.__randomizer = randomizer
    self.__split = split
    self.__writer = writer

  def _interpret_label_features(self, bytes_features, float_features,
                                int_features):
    """ This method takes the output feature list from the frame randomizer, and
    converts them into a dictionary with the following keys:
      dots: The dot position feature.
      face_size: The face size feature.
      leye_box: The left eye bounding box feature.
      reye_box: The right eye bounding box feature.
      grid_box: The face grid feature.
      session_num: The session number feature.
      pose: The head pose feature. (optional)
    Args:
      bytes_features: The list of bytes features.
      float_features: The list of float features.
      int_features: The list of int features.
    Returns:
      The specified dictionary. """
    raise NotImplementedError( \
        "_interpret_label_features() must be implemented by subclass.")

  def save_all(self):
    """ Saves all the images from the frame randomizer. """
    i = 0
    last_percentage = 0.0
    while True:
      try:
        frame, bytes_f, float_f, int_f = self.__randomizer.get_random_example()
      except ValueError as e:
        # No more frames to read.
        break

      # Calculate percentage complete.
      percent = float(i) / self.__randomizer.get_num_examples() * 100
      if percent - last_percentage > 0.01:
        print "Processing %s split. (%.2f%% done)" % (split, percent)
        last_percentage = percent

      # Convert the label features.
      features = self._interpret_label_features(bytes_f, float_f, int_f)

      # Create the combined feature.
      split = self.__split
      combined_feature = {"%s/dots" % (split): features["dots"],
                          "%s/face_size" % (split): features["face_size"],
                          "%s/leye_box" % (split): features["leye_box"],
                          "%s/reye_box" % (split): features["reye_box"],
                          "%s/grid_box" % (split): features["grid_box"],
                          "%s/session_num" % (split): features["session_num"],
                          "%s/image" % (split): frame}
      if "pose" in features:
        # Include head pose.
        combined_feature["%s/pose" % (split)] = features["pose"]
      example = \
          tf.train.Example(features=tf.train.Features(feature=combined_feature))

      # Write it out.
      self.__writer.write(example.SerializeToString())

      i += 1

  def close(self):
    """ Closes internal writer. """
    return self.__writer.close()
