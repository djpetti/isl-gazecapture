import keras.backend as K
import keras.layers as layers

from network import Network


class Autoencoder(Network):
  """ Implements autoencoder for analysing variations in eye or face appearance.
  """

  def _build_custom(self):
    trainable = not self._fine_tune

    pool_eye_in = layers.MaxPooling2D(pool_size=2, padding="same")

    # Encoder layers.
    #conv_e1 = layers.Conv2D(48, (11, 11), strides=(2, 2), activation="relu",
    #                        padding="same",
    #                        kernel_regularizer=self._l2, trainable=trainable)
    #norm_e1 = layers.BatchNormalization(trainable=trainable)

    conv_e2 = layers.Conv2D(16, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e2 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e2 = layers.BatchNormalization(trainable=trainable)

    conv_e3 = layers.Conv2D(8, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e3 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e3 = layers.BatchNormalization(trainable=trainable)

    conv_e4 = layers.Conv2D(8, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e4 = layers.MaxPooling2D(pool_size=2, padding="same")
    norm_e4 = layers.BatchNormalization(trainable=trainable)

    # Decoder layers.
    conv_d1 = layers.Conv2D(8, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    upsample_d1 = layers.UpSampling2D(size=2)
    norm_d1 = layers.BatchNormalization(trainable=trainable)

    conv_d2 = layers.Conv2D(8, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    upsample_d2 = layers.UpSampling2D(size=2)
    norm_d2 = layers.BatchNormalization(trainable=trainable)

    conv_d3 = layers.Conv2D(16, (3, 3), activation="relu",
                            padding="same",
                            kernel_regularizer=self._l2, trainable=trainable)
    upsample_d3 = layers.UpSampling2D(size=2)
    norm_d3 = layers.BatchNormalization(trainable=trainable)

    conv_d4 = layers.Conv2D(1, (3, 3), padding="same",
                            kernel_regularizer=self._l2, trainable=trainable,
                            name="decode")

    #norm_d4 = layers.BatchNormalization(trainable=trainable)
    #upsample_d4 = layers.UpSampling2D(size=2)
    #conv_d4 = layers.Conv2D(48, (11, 11), activation="relu",
    #                        padding="same",
    #                        kernel_regularizer=self._l2, trainable=trainable,
    #                        name="decode")

    self._small_eye = pool_eye_in(self._left_eye_node)

    # Build the autoencoder network.
    #enc1 = conv_e1(self._small_eye)
    #enc2 = norm_e1(enc1)
    enc3 = conv_e2(self._small_eye)
    enc4 = pool_e2(enc3)
    enc5 = norm_e2(enc4)
    enc6 = conv_e3(enc5)
    enc7 = pool_e3(enc6)
    enc8 = norm_e3(enc7)
    enc9 = conv_e4(enc8)
    enc10 = pool_e4(enc9)
    enc11 = norm_e4(enc10)

    dec1 = conv_d1(enc11)
    dec2 = upsample_d1(dec1)
    dec3 = norm_d1(dec2)
    dec4 = conv_d2(dec3)
    dec5 = upsample_d2(dec4)
    dec6 = norm_d2(dec5)
    dec7 = conv_d3(dec6)
    dec8 = upsample_d3(dec7)
    dec9 = norm_d3(dec8)
    dec10 = conv_d4(dec9)

    # Build the gaze prediction pathway.
    self.__encoded = layers.Flatten(name="encode")(enc11)
    gaze_dense1 = layers.Dense(128, activation="relu",
                               kernel_regularizer=self._l2,
                               trainable=trainable)(self.__encoded)
    gaze_dense2 = layers.Dense(128, activation="relu",
                               kernel_regularizer=self._l2,
                               trainable=trainable)(gaze_dense1)
    gaze_pred = layers.Dense(2, kernel_regularizer=self._l2,
                             trainable=trainable,
                             name="dots")(gaze_dense2)

    # The outputs are the decoded input and the gaze prediction.
    return dec10, gaze_pred, self.__encoded

  def prepare_labels(self, dots):
    """ We abuse the prepare_labels functionality a little so that we can get
    the right label data for this network without having to mess with the
    experiment code.
    Args:
      dots: The input dots feature.
    Returns:
      The decodings and gaze predictions. """
    # The expected decoding is just the input.
    labels = dots.copy()
    labels["decode"] = self._small_eye
    # We don't really care about the encoded representation, so we can just set
    # the labels to what the output already is.
    labels["encode"] = self.__encoded
    return labels
