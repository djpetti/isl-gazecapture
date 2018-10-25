def _set_trainable(model, set_to):
  """ Sets the trainable attribute of all layers in the model.
  Args:
    model: The model to modify.
    set_to: What to set the trainable property to. """
  for layer in model.layers:
    layer.trainable = set_to

def freeze_all(model):
  """ Freezes all the layers in the model.
  Args:
    model: The model to freeze. """
  _set_trainable(model, False)

def unfreeze_all(model):
  """ Unfreezes all the layers in the model.
  Args:
    model: The model to unfreeze. """
  _set_trainable(model, True)
