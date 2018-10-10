from network import Network

import keras.layers as layers


class Autoencoder(Network):
  """ Implements autoencoder for analysing variations in eye or face appearance.
  """

  def _build_custom(self):
    trainable = not self._fine_tune

    # Encoder layers.
    conv_e1 = layers.Conv2D(48, (11, 11), strides=(4, 4), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e1 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e1 = layers.BatchNormalization(trainable=trainable)

    conv_e2 = layers.Conv2D(128, (5, 5), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e2 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e2 = layers.BatchNormalization(trainable=trainable)

    conv_e3 = layers.Conv2D(192, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e3 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e3 = layers.BatchNormalization(trainable=trainable)

    conv_e4 = layers.Conv2D(192, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e4 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e4 = layers.BatchNormalization(trainable=trainable)

    conv_e5 = layers.Conv2D(192, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e5 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e5 = layers.BatchNormalization(trainable=trainable)

    conv_e6 = layers.Conv2D(32, (1, 1), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    # Decoder layers.
    conv_d1 = layers.Conv2D(32, (1, 1), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    norm_d2 = layers.BatchNormalization(trainable=trainable)
    upsample_d2 = layers.UpSampling2D(size=2)
    conv_d2 = layers.Conv2D(192, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    norm_d3 = layers.BatchNormalization(trainable=trainable)
    upsample_d3 = layers.UpSampling2D(size=2)
    conv_d3 = layers.Conv2D(192, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    norm_d4 = layers.BatchNormalization(trainable=trainable)
    upsample_d4 = layers.UpSampling2D(size=2)
    conv_d4 = layers.Conv2D(192, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    norm_d5 = layers.BatchNormalization(trainable=trainable)
    upsample_d5 = layers.UpSampling2D(size=2)
    conv_d5 = layers.Conv2D(128, (5, 5), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    norm_d6 = layers.BatchNormalization(trainable=trainable)
    upsample_d6 = layers.UpSampling2D(size=2)
    conv_d6 = layers.Conv2D(48, (11, 11), strides=(4, 4), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)

    # Build the autoencoder network.
    enc1 = conv_e1(self._left_eye_node)
    enc2 = pool_e1(enc1)
    enc3 = norm_e1(enc2)
    enc4 = conv_e2(enc3)
    enc5 = pool_e2(enc4)
    enc6 = norm_e2(enc5)
    enc7 = conv_e3(enc6)
    enc8 = pool_e3(enc7)
    enc9 = norm_e3(enc8)
    enc10 = conv_e4(enc9)
    enc11 = pool_e4(enc10)
    enc12 = norm_e4(enc11)
    enc13 = conv_e5(enc12)
    enc14 = pool_e5(enc13)
    enc15 = norm_e5(enc14)
    enc16 = conv_e6(enc15)

    dec1 = conv_d1(enc16)
    dec2 = norm_d2(dec1)
    dec3 = upsample_d2(dec2)
    dec4 = conv_d2(dec3)
    dec5 = norm_d3(dec4)
    dec6 = upsample_d3(dec5)
    dec7 = conv_d3(dec6)
    dec8 = norm_d4(dec7)
    dec9 = upsample_d4(dec8)
    dec10 = conv_d4(dec9)
    dec11 = norm_d5(dec10)
    dec12 = upsample_d5(dec11)
    dec13 = conv_d5(dec12)
    dec14 = norm_d6(dec13)
    dec15 = upsample_d6(dec14)
    dec16 = conv_d6(dec15)

    # Build the gaze prediction pathway.
    encoded = layers.Flatten()(enc16)
    gaze_dense1 = layers.Dense(128, activation="relu",
                               kernel_regularizer=self._l2,
                               trainable=trainable)(encoded)
    gaze_pred = layers.Dense(2, activation="relu",
                             kernel_regularizer=self._l2,
                             trainable=trainable)(gaze_dense1)

    # The outputs are the decoded input and the gaze prediction.
    return dec16, gaze_pred

  def prepare_labels(self, dots):
    """ We abuse the prepare_labels functionality a little so that we can get
    the right label data for this network without having to mess with the
    experiment code.
    Args:
      dots: The input dots feature.
    Returns:
      The decodings and gaze predictions. """
    # The expected decoding is just the input.
    return [self._left_eye_node] + dots
