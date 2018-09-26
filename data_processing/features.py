import tensorflow as tf


def to_ints(value):
	""" Converts a list to an int64 feature.
	Args:
		value: The list to convert.
	Returns:
	  The corresponding feature. """
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def to_bytes(value):
  """ Converts a list to a uint8 feature.
  Args:
    value: The list to convert.
  Returns:
    The corresponding feature. """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def to_floats(value):
  """ Converts a list to a float32 feature.
  Args:
    value: The list to convert.
  Returns:
    The corresponding feature. """
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

