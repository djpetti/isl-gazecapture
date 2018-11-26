import tensorflow as tf

from network import Network


layers = tf.keras.layers
applications = tf.keras.applications


class ResidualNetwork(Network):
  """ Uses residual-module-based architecture for eye feature extraction. """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Get pretrained VGG model for use as a base.
    vgg = applications.vgg19.VGG19(include_top=False,
                                   input_shape=self._input_shape)
    vgg_out = vgg(self._face_node)

    # Freeze all layers in VGG.
    for layer in vgg.layers:
      layer.trainable = False

    # Shared eye layers.
    conv1_e1 = layers.Conv2D(96, (11, 11), strides=(4, 4), activation="relu",
                             kernel_regularizer=self._l2)
    conv1_e2 = layers.Conv2D(96, (1, 1), activation="relu",
                             kernel_regularizer=self._l2)
    norm1 = layers.BatchNormalization(trainable=trainable)

    pool1 = layers.MaxPooling2D(pool_size=3, strides=(2, 2))

    pad2 = layers.ZeroPadding2D(padding=(2, 2))
    conv2_e1 = layers.Conv2D(171, (3, 3), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv2_e2 = layers.Conv2D(171, (3, 3), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv2_e3 = layers.Conv2D(171, (1, 1), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)

    shortcut2 = layers.Conv2D(171, (1, 1), activation="relu",
                              kernel_regularizer=self._l2, trainable=trainable)
    norm2 = layers.BatchNormalization(trainable=trainable)

    pool2 = layers.MaxPooling2D(pool_size=3, strides=(2, 2))

    pad3 = layers.ZeroPadding2D(padding=(1, 1))
    conv3_e1 = layers.Conv2D(256, (3, 3), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv3_e2 = layers.Conv2D(256, (1, 1), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)
    conv3_e3 = layers.Conv2D(256, (1, 1), activation="relu",
                             kernel_regularizer=self._l2, trainable=trainable)

    shortcut3 = layers.Conv2D(256, (1, 1), activation="relu",
                              kernel_regularizer=self._l2, trainable=trainable)
    norm3 = layers.BatchNormalization(trainable=trainable)

    conv4_e1 = layers.Conv2D(64, (1, 1), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    flatten4 = layers.Flatten()

    # Left eye stack.
    leye_conv1_e1 = conv1_e1(self._left_eye_node)
    leye_conv1_e2 = conv1_e2(leye_conv1_e1)
    #leye_norm1 = norm1(leye_conv1_e2)
    leye_pool1 = pool1(leye_conv1_e2)

    leye_pad2 = pad2(leye_pool1)
    leye_conv2_e1 = conv2_e1(leye_pad2)
    leye_conv2_e2 = conv2_e2(leye_conv2_e1)
    leye_conv2_e3 = conv2_e3(leye_conv2_e2)
    leye_shortcut2 = shortcut2(leye_pool1)
    leye_mod2 = layers.Add()([leye_conv2_e3, leye_shortcut2])
    #leye_norm2 = norm2(leye_mod2)
    leye_pool2 = pool2(leye_mod2)

    leye_pad3 = pad3(leye_pool2)
    leye_conv3_e1 = conv3_e1(leye_pad3)
    leye_conv3_e2 = conv3_e2(leye_conv3_e1)
    leye_conv3_e3 = conv3_e3(leye_conv3_e2)
    leye_shortcut3 = shortcut3(leye_pool2)
    leye_mod3 = layers.Add()([leye_conv3_e3, leye_shortcut3])
    #leye_norm3 = norm3(leye_mod3)

    leye_conv4_e1 = conv4_e1(leye_mod3)
    leye_flatten4 = flatten4(leye_conv4_e1)

    # Right eye stack.
    reye_conv1_e1 = conv1_e1(self._left_eye_node)
    reye_conv1_e2 = conv1_e2(reye_conv1_e1)
    #reye_norm1 = norm1(reye_conv1_e2)
    reye_pool1 = pool1(reye_conv1_e2)

    reye_pad2 = pad2(reye_pool1)
    reye_conv2_e1 = conv2_e1(reye_pad2)
    reye_conv2_e2 = conv2_e2(reye_conv2_e1)
    reye_conv2_e3 = conv2_e3(reye_conv2_e2)
    reye_shortcut2 = shortcut2(reye_pool1)
    reye_mod2 = layers.Add()([reye_conv2_e3, reye_shortcut2])
    #reye_norm2 = norm2(reye_mod2)
    reye_pool2 = pool2(reye_mod2)

    reye_pad3 = pad3(reye_pool2)
    reye_conv3_e1 = conv3_e1(reye_pad3)
    reye_conv3_e2 = conv3_e2(reye_conv3_e1)
    reye_conv3_e3 = conv3_e3(reye_conv3_e2)
    reye_shortcut3 = shortcut3(reye_pool2)
    reye_mod3 = layers.Add()([reye_conv3_e3, reye_shortcut3])
    #reye_norm3 = norm3(reye_mod3)

    reye_conv4_e1 = conv4_e1(reye_mod3)
    reye_flatten4 = flatten4(reye_conv4_e1)

    # Concatenate eyes and put through a shared FC layer.
    eye_combined = layers.Concatenate()([reye_flatten4, leye_flatten4])
    eye_drop = layers.Dropout(0.5)(eye_combined)
    fc_e1 = layers.Dense(128, activation="relu",
                        kernel_regularizer=self._l2)(eye_drop)

    # Face layers.
    face_flatten_f4 = layers.Flatten()(vgg_out)

    face_drop = layers.Dropout(0.5)(face_flatten_f4)
    face_fc1 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(face_drop)
    face_fc2 = layers.Dense(64, activation="relu",
                            kernel_regularizer=self._l2)(face_fc1)

    # Face grid.
    grid_flat = layers.Flatten()(self._grid_input)
    grid_fc1 = layers.Dense(256, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_flat)
    grid_fc2 = layers.Dense(128, activation="relu",
                            kernel_regularizer=self._l2,
                            trainable=trainable)(grid_fc1)

    # Concat everything and put through a final FF layer.
    all_concat = layers.Concatenate()([fc_e1, face_fc2, grid_fc2])
    all_fc1 = layers.Dense(128, activation="relu",
                          kernel_regularizer=self._l2)(all_concat)
    all_fc2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1)

    return all_fc2

