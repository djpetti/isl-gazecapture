from custom_layers.residual import ResNetBlock

import keras.layers as layers

from network import Network


class ResNetwork(Network):
  """ A ResNet-based gaze network. """

  def __build_group(self, num_blocks, *args, **kwargs):
    """ Creates a group of ResNetBlocks.
    Args:
      num_blocks: The number of blocks in the group.
      no_downsample: By default, it downsamples on the first layer. If set to
                     true, it won't do this.
      All other arguments will be passed transparently to the ResNetBlock
      constructor.
    Returns:
      A list of the blocks that it created. """
    downsample = not kwargs.get("no_downsample", False)
    if "no_downsample" in kwargs:
      kwargs.pop("no_downsample")

    # Downsample on the first one.
    block = ResNetBlock(*args, downsample_first=downsample, **kwargs)
    blocks = [block]

    # Add the rest.
    for i in range(1, num_blocks):
      block = ResNetBlock(*args, **kwargs)
      blocks.append(block)

    return blocks

  def __apply_all(self, inputs, layers):
    """ Creates a stack of layers, where each layer is applied to the output of
    the previous one.
    Args:
      inputs: The initial inputs to apply the stack on.
      layers: The layers to apply.
    Returns:
      The output from the last layer. """
    next_input = inputs

    for layer in layers:
      next_input = layer(next_input)

    return next_input

  def _build_custom(self):
    trainable = not self._fine_tune

    # Shared eye layers.
    conv_e1 = layers.Conv2D(64, (7, 7), strides=(2, 2),
                            padding="same", kernel_regularizer=self._l2,
                            trainable=trainable)
    norm_e1 = layers.BatchNormalization()
    act_e1 = layers.Activation("relu")
    pool_e1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")

    res_e2 = self.__build_group(3, 64, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable,
                                no_downsample=True)
    res_e3 = self.__build_group(4, 128, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)
    res_e4 = self.__build_group(6, 256, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)
    res_e5 = self.__build_group(3, 512, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)

    conv_e6 = layers.Conv2D(64, (1, 1), kernel_regularizer=self._l2,
                            trainable=trainable)
    norm_e6 = layers.BatchNormalization()
    act_e6 = layers.Activation("relu")

    eye_layers = [conv_e1, act_e1, norm_e1, pool_e1] + res_e2 + res_e3 + \
                 res_e4 + res_e5 + [conv_e6, norm_e6, act_e6]
    # Left eye stack.
    leye_out = self.__apply_all(self._left_eye_node, eye_layers)
    # Right eye stack.
    reye_out = self.__apply_all(self._right_eye_node, eye_layers)

    # Concatenate eyes.
    eye_concat = layers.Concatenate()([leye_out, reye_out])

    # Operate on combined eyes.
    res_ec1 = self.__build_group(3, 128, (3, 3), activation="relu",
                                 kernel_regularizer=self._l2,
                                 trainable=trainable,
                                 no_downsample=True)
    res_ec2 = self.__build_group(3, 256, (3, 3), activation="relu",
                                 kernel_regularizer=self._l2,
                                 trainable=trainable,
                                 no_downsample=True)

    conv_ec3 = layers.Conv2D(128, (1, 1),
                             kernel_regularizer=self._l2,
                             trainable=trainable)
    norm_ec3 = layers.BatchNormalization()
    act_ec3 = layers.Activation("relu")

    combined_eye_layers = res_ec1 + res_ec2 + [conv_ec3, norm_ec3, act_ec3]
    eye_comb_out = self.__apply_all(eye_concat, combined_eye_layers)

    # Face layers.
    conv_f1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    norm_f1 = layers.BatchNormalization()
    act_f1 = layers.Activation("relu")
    pool_f1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")

    res_f2 = self.__build_group(3, 64, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable,
                                no_downsample=True)
    res_f3 = self.__build_group(4, 128, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)
    res_f4 = self.__build_group(6, 256, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)
    res_f5 = self.__build_group(3, 512, (3, 3), activation="relu",
                                kernel_regularizer=self._l2,
                                trainable=trainable)

    conv_f6 = layers.Conv2D(64, (1, 1),
                            kernel_regularizer=self._l2, trainable=trainable)
    norm_f6 = layers.BatchNormalization()
    act_f6 = layers.Activation("relu")

    # Face stack.
    face_layers = [conv_f1, norm_f1, act_f1, pool_f1] + res_f2 + res_f3 + \
                  res_f4 + res_f5 + [conv_f6, norm_f6, act_f6]
    face_out = self.__apply_all(self._face_node, face_layers)

    # Face grid.
    grid_flat = layers.Flatten()
    grid_fg1 = layers.Dense(256, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)
    norm_fg1 = layers.BatchNormalization()
    grid_fg2 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)
    norm_fg2 = layers.BatchNormalization()

    grid_out = self.__apply_all(self._grid_input,
                                [grid_flat, grid_fg1, norm_fg1,
                                 grid_fg2, norm_fg2])

    # Concat everything and put it through a final dense stack.
    eye_out_flat = layers.Flatten()(eye_comb_out)
    face_out_flat = layers.Flatten()(face_out)

    all_concat = layers.Concatenate()([eye_out_flat, face_out_flat, grid_out])

    all_fc1 = layers.Dense(128, activation="relu",
                           kernel_regularizer=self._l2,
                           trainable=trainable)
    norm_fc1 = layers.BatchNormalization()
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2, name="dots")

    all_out = self.__apply_all(all_concat, [all_fc1, norm_fc1, all_fc2])
    return all_out
