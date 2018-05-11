import keras.backend as K

import tensorflow as tf


def fuse_loaders(train_outputs, test_outputs):
  """ Keras is capable of loading data and labels from custom tensors. However,
  it does not support toggling between training and testing inputs. Therefore,
  this method uses a conditional operation to select between the two loaders.
  Args:
    train_outputs: The output tensors from the training data loader.
    test_outputs: The output tensors from the testing data loader.
  Returnes:
    The output tensors. """
  is_train = K.learning_phase()
  return tf.cond(is_train, lambda: train_outputs, lambda: test_outputs)
