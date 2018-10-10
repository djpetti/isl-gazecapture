import keras.layers as layers
import keras.applications as applications

import tensorflow as tf

from network import Network


class BranchedNetwork(Network):
  """ Extension of LargeVggNetwork that uses a branched architecture based on
  the size of the subject's face. """

  # Cutoff face area for determining which group an image belongs to. In this
  # case, it is simply the mean area value.
  _FACE_AREA_CUTOFF = 127

  def __get_face_groups(self, inputs):
    """ Determines which group to put the image in based on the face size.
    Args:
      inputs: Following parameters as a list:
        face_mask: Input vector of face masks.
        to_branch: A corresponding vector that we want to separate into groups.
    Returns:
      Tensor containing images belonging to the large group, and a tensor
      containing images belonging to the small group. """
    face_mask, to_branch = inputs

    # Determine the face area.
    face_area = tf.count_nonzero(face_mask, axis=[1, 2])

    # Create masks for each group.
    large_mask = tf.greater(face_area, self._FACE_AREA_CUTOFF)
    small_mask = tf.logical_not(large_mask)

    # Split each group.
    large_group = tf.boolean_mask(to_branch, large_mask)
    small_group = tf.boolean_mask(to_branch, small_mask)

    return [large_group, small_group]

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
    conv_e1 = layers.Conv2D(144, (11, 11), strides=(4, 4), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e1 = layers.BatchNormalization(trainable=trainable)

    pad_e2 = layers.ZeroPadding2D(padding=(2, 2))
    conv_e2 = layers.Conv2D(384, (5, 5), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    pool_e2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
    norm_e2 = layers.BatchNormalization(trainable=trainable)

    pad_e3 = layers.ZeroPadding2D(padding=(1, 1))
    conv_e3 = layers.Conv2D(576, (3, 3), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)

    conv_e4 = layers.Conv2D(64, (1, 1), activation="relu",
                            kernel_regularizer=self._l2, trainable=trainable)
    flatten_e4 = layers.Flatten()

    # Left eye stack.
    leye_conv_e1 = conv_e1(self._left_eye_node)
    leye_pool_e1 = pool_e1(leye_conv_e1)
    leye_norm_e1 = norm_e1(leye_pool_e1)

    leye_pad_e2 = pad_e2(leye_norm_e1)
    leye_conv_e2 = conv_e2(leye_pad_e2)
    leye_pool_e2 = pool_e2(leye_conv_e2)
    leye_norm_e2 = norm_e2(leye_pool_e2)

    leye_pad_e3 = pad_e3(leye_norm_e2)
    leye_conv_e3 = conv_e3(leye_pad_e3)

    leye_conv_e4 = conv_e4(leye_conv_e3)
    leye_flatten_e4 = flatten_e4(leye_conv_e4)

    # Right eye stack.
    reye_conv_e1 = conv_e1(self._right_eye_node)
    reye_pool_e1 = pool_e1(reye_conv_e1)
    reye_norm_e1 = norm_e1(reye_pool_e1)

    reye_pad_e2 = pad_e2(reye_norm_e1)
    reye_conv_e2 = conv_e2(reye_pad_e2)
    reye_pool_e2 = pool_e2(reye_conv_e2)
    reye_norm_e2 = norm_e2(reye_pool_e2)

    reye_pad_e3 = pad_e3(reye_norm_e2)
    reye_conv_e3 = conv_e3(reye_pad_e3)

    reye_conv_e4 = conv_e4(reye_conv_e3)
    reye_flatten_e4 = flatten_e4(reye_conv_e4)

    # Concatenate eyes and put through a shared FC layer.
    eye_combined = layers.Concatenate()([reye_flatten_e4, leye_flatten_e4])
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

    # Create a special layer for computing the branching.
    branch_layer = layers.Lambda(self.__get_face_groups)
    grid_fc2_b1, grid_fc2_b2 = branch_layer([self._grid_input, grid_fc2])
    # We have to use the branch layer on everything we want to concatenate so
    # they have compatible input shapes.
    fc_e1_b1, fc_e1_b2 = branch_layer([self._grid_input, fc_e1])
    face_fc2_b1, face_fc2_b2 = branch_layer([self._grid_input, face_fc2])

    # Concat everything and put through final FF layers.
    all_concat_b1 = layers.Concatenate()([fc_e1_b1, face_fc2_b1, grid_fc2_b1])
    all_fc1_b1 = layers.Dense(128, activation="relu",
                              kernel_regularizer=self._l2)(all_concat_b1)
    all_fc2_b1 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1_b1)

    # Same for the second branch.
    all_concat_b2 = layers.Concatenate()([fc_e1_b2, face_fc2_b2, grid_fc2_b2])
    all_fc1_b2 = layers.Dense(128, activation="relu",
                              kernel_regularizer=self._l2)(all_concat_b2)
    all_fc2_b2 = layers.Dense(2, kernel_regularizer=self._l2)(all_fc1_b2)

    # For training, we can now concatenate the output from the two branches back
    # together.
    all_fc2 = layers.Concatenate(axis=0)([all_fc2_b1, all_fc2_b2])

    return all_fc2

  def prepare_labels(self, labels):
    # Since we reorder the output from our model relative to the input, we need
    # our labels to be organized similarly.
    labels_group1, labels_group2 = \
        self.__get_face_groups((self._grid_input, labels))
    return tf.concat([labels_group1, labels_group2], axis=0)

