import cPickle as pickle
import logging

import keras.layers as layers
import keras.applications as applications

import tensorflow as tf

from .. import utils
from network import Network
import autoencoder


logger = logging.getLogger(__name__)


class BranchedAutoencNetwork(Network):
  """ Extension of LargeVggNetwork that uses a branched architecture based on
  the appearance of a subject's eye. """

  def __init__(self, *args, **kwargs):
    """ Takes the same parameters as Network. Additionally, it requires the
    following parameters:
    Args:
      autoenc_model_file: The saved weights to use for the autoencoder.
      cluster_data: The file containing the saved clustering data. """
    self.__autoenc_file = kwargs.get("autoenc_model_file")
    if not self.__autoenc_file:
      raise ValueError("'autoenc_model_file' arg is required.")
    self.__cluster_file = kwargs.get("cluster_data")
    if not self.__cluster_file:
      raise ValueError("'cluster_data' arg is required.")

    self.__autoencoder = None
    self.__clusters = None

    super(BranchedAutoencNetwork, self).__init__(*args, **kwargs)

  def __load_autoencoder(self):
    """ Loads the autoencoder model. """
    net = autoencoder.Autoencoder(self._input_shape, eye_shape=self._eye_shape)
    self.__autoencoder = net.build()

    # Load the saved weights.
    logger.info("Loading autoencoder weights from %s." % (self.__autoenc_file))
    self.__autoencoder.load_weights(self.__autoenc_file)

    # Load the cluster data.
    logger.info("Loading clusters from %s." % (self.__cluster_file))
    cluster_file = file(self.__cluster_file, "rb")
    self.__clusters = pickle.load(cluster_file)

    # Freeze the model.
    utils.freeze_all(self.__autoencoder)

  def __compute_groups(self, encodings):
    """ Computes the groups given encodings based on cluster data.
    Args:
      encodings: The encodings we want to group.
    Returns:
      A vector with the same first dimension as encodings in which each element
      signifies the chosen group for the corresponding encoding. """
    def distance(center):
      """ Computes the euclidean distances between a set of encodings and a
      cluster center.
      Args:
        center: The cluster center.
      Returns:
        A vector of the corresponding distances. """
      return tf.norm(encodings - center, axis=1)

    # Compute distance to all the cluster centroids.
    center_distances = []
    for center in self.__clusters:
      center_dist = distance(center)
      center_distances.append(center_dist)

    # Figure out the closest one for each encoding.
    distances = tf.stack(center_distances, axis=0)
    closest = tf.argmin(distances, axis=0)

    return closest

  def __get_appearance_groups(self, to_branch):
    """ Determines which group to put the image in based on the appearance.
    Args:
      to_branch: A vector corresponding to the left eye inputs that we want
                 to separate into groups.
    Returns:
      Tensor containing images belonging to the first group, and a tensor
      containing images belonging to the second group. """
    if not self.__autoencoder:
      # We need to load the autoencoder model if we haven't done so.
      self.__load_autoencoder()

    # Compute encodings for each group.
    _, _, encodings = self.__autoencoder([self._left_eye_node,
                                          self._right_eye_node,
                                          self._face_node,
                                          self._grid_input])

    # Compute the groups based on the encodings.
    groups = self.__compute_groups(encodings)

    # Create masks for each group.
    zeros = tf.zeros_like(groups)
    first_mask = tf.equal(groups, zeros)
    second_mask = tf.logical_not(first_mask)

    # Split each group.
    first_group = tf.boolean_mask(to_branch, first_mask)
    second_group = tf.boolean_mask(to_branch, second_mask)

    return [first_group, second_group]

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
    branch_layer = layers.Lambda(self.__get_appearance_groups)
    grid_fc2_b1, grid_fc2_b2 = branch_layer(grid_fc2)
    # We have to use the branch layer on everything we want to concatenate so
    # they have compatible input shapes.
    fc_e1_b1, fc_e1_b2 = branch_layer(fc_e1)
    face_fc2_b1, face_fc2_b2 = branch_layer(face_fc2)

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
    all_fc2 = layers.Concatenate(axis=0, name="dots")([all_fc2_b1, all_fc2_b2])

    return all_fc2

  def prepare_labels(self, labels):
    # Since we reorder the output from our model relative to the input, we need
    # our labels to be organized similarly.
    labels_group1, labels_group2 = \
        self.__get_appearance_groups(labels["dots"])
    dots = tf.concat([labels_group1, labels_group2], axis=0)

    return {"dots": dots}
