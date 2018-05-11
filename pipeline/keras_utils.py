import tensorflow as tf


def fuse_loaders(loader1_outputs, loader2_outputs):
  """ Keras is capable of loading data and labels from custom tensors. However,
  it does not support toggling between training and testing inputs. Therefore,
  this method uses a conditional operation to select between the two loaders,
  and provides a special placeholder to control whether we're training or
  testing.
  Args:
    loader1_outputs: The output tensors from the first data loader.
    loader2_outputs: The output tensors from the second data loader.
  Returnes:
    The output tensors and selection placeholder. The selection
    placeholder should be fed with True when testing. """
  # Create selection placeholder.
  is_testing = tf.placeholder(tf.bool, shape=[])
  pred = tf.equal(is_testing, True)

  fused = tf.cond(pred, lambda: [loader1_outputs],
                        lambda: [loader2_outputs])

  return (fused, is_testing)
