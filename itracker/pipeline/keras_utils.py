import tensorflow as tf


K = tf.keras.backend


def fuse_loaders(train_outputs, test_outputs):
  """ Keras is capable of loading data and labels from custom tensors. However,
  it does not support toggling between training and testing inputs. Therefore,
  this method uses a conditional operation to select between the two loaders.
  Args:
    train_outputs: The output tensors from the training data loader.
    test_outputs: The output tensors from the testing data loader.
  Returnes:
    The output tensors. """
  fused_outputs = []
  # We have to combine them on a tensor-by-tensor basis.
  for train_output, test_output in zip(train_outputs, test_outputs):
    fused_output = K.in_train_phase(train_output, test_output)
    fused_outputs.append(fused_output)

  return fused_outputs


def pipeline_input(*args, **kwargs):
  """ Input layer specialization for use with Pipeline outputs. """
  if kwargs.get("tensor") is None:
    # Tensor is a required argument for this subclass.
    raise ValueError("'tensor' argument is required.")

  # Create the base layer.
  input_tensor = tf.keras.Input(*args, **kwargs)

  # Hack for ensuring that Keras recognizes that the pipeline system depends
  # on the learning_phase placeholder.
  input_tensor._uses_learning_phase = True

  return input_tensor
