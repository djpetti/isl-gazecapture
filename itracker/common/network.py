from keras.models import Model, load_model
import keras.backend as K
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers

import tensorflow as tf


# Shape of the inputs to the network.
INPUT_SHAPE = (224, 224, 3)


def rgb_to_grayscale(image_tensor):
  """ Converts a tensor of RGB images to grayscale. This is meant to be used in
  a Keras Lambda layer.
  Args:
    image_tensor: The tensor of images to convert.
  Returns:
    The same tensor, with all images converted to grayscale. """
  # Weight each channel before averaging.
  luma_weights = tf.constant([[0.21], [0.72], [0.07]])
  # Average using a single contraction operation.
  grayscale = tf.tensordot(image_tensor, luma_weights, axes=[[3], [0]])

  return grayscale

def build_network(face_shape=None, fine_tune=False):
  """ Builds the network.
  Args:
    face_shape: If specified, the network can automatically resize the face
                input.
    fine_tune: Whether we are fine-tuning the model. If so, only the last two
               layers will be trainable.
  Returns:
    The built network, ready to train. """
  trainable = not fine_tune

  # L2 regularizer for weight decay.
  l2_reg = regularizers.l2(0.0005)

  left_eye_input = layers.Input(shape=INPUT_SHAPE, name="left_eye_input")
  right_eye_input = layers.Input(shape=INPUT_SHAPE, name="right_eye_input")
  # The face crop gets resized on-the-fly.
  if face_shape is None:
    # Expect the input shape.
    face_shape = INPUT_SHAPE
  face_input = layers.Input(shape=face_shape, name="face_input")
  grid_input = layers.Input(shape=(25, 25), name="grid_input")

  left_eye_floats = K.cast(left_eye_input, "float32")
  right_eye_floats = K.cast(right_eye_input, "float32")
  face_floats = K.cast(face_input, "float32")
  grid_floats = K.cast(grid_input, "float32")

  face_scaled = face_floats
  if face_shape != INPUT_SHAPE:
    # Resize face.
    scale_layer = layers.Lambda(lambda x: \
                                tf.image.resize_images(x, (INPUT_SHAPE[0],
                                                           INPUT_SHAPE[1])))
    face_scaled = scale_layer(face_floats)

  # Convert everything to grayscale.
  gray_layer = layers.Lambda(lambda x: rgb_to_grayscale(x))
  left_eye_gray = gray_layer(left_eye_floats)
  right_eye_gray = gray_layer(right_eye_floats)
  face_gray = gray_layer(face_scaled)

  # Shared eye layers.
  conv_e1 = layers.Conv2D(144, (11, 11), strides=(4, 4), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)
  pool_e1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
  norm_e1 = layers.BatchNormalization(trainable=trainable)

  pad_e2 = layers.ZeroPadding2D(padding=(2, 2))
  conv_e2 = layers.Conv2D(384, (5, 5), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)
  pool_e2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
  norm_e2 = layers.BatchNormalization(trainable=trainable)

  pad_e3 = layers.ZeroPadding2D(padding=(1, 1))
  conv_e3 = layers.Conv2D(576, (3, 3), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)

  conv_e4 = layers.Conv2D(64, (1, 1), activation="relu",
                          kernel_regularizer=l2_reg, trainable=trainable)
  flatten_e4 = layers.Flatten()

  # Left eye stack.
  leye_conv_e1 = conv_e1(left_eye_gray)
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
  reye_conv_e1 = conv_e1(right_eye_gray)
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
  fc_e1 = layers.Dense(128, activation="relu",
                       kernel_regularizer=l2_reg,
                       trainable=trainable)(eye_combined)

  # Face layers.
  face_conv_f1 = layers.Conv2D(144, (11, 11), strides=(4, 4),
                               activation="relu",
                               kernel_regularizer=l2_reg,
                               trainable=trainable)(face_gray)
  face_pool_f1 = layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2))(face_conv_f1)
  face_norm_f1 = layers.BatchNormalization(trainable=trainable)(face_pool_f1)

  face_pad_f2 = layers.ZeroPadding2D(padding=(2, 2))(face_norm_f1)
  face_conv_f2 = layers.Conv2D(384, (5, 5), activation="relu",
                               kernel_regularizer=l2_reg,
                               trainable=trainable)(face_pad_f2)
  face_pool_f2 = layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2))(face_conv_f2)
  face_norm_f2 = layers.BatchNormalization(trainable=trainable)(face_pool_f2)

  face_pad_f3 = layers.ZeroPadding2D(padding=(1, 1))(face_norm_f2)
  face_conv_f3 = layers.Conv2D(576, (3, 3), activation="relu",
                               kernel_regularizer=l2_reg,
                               trainable=trainable)(face_pad_f3)

  face_conv_f4 = layers.Conv2D(64, (1, 1), activation="relu",
                               kernel_regularizer=l2_reg,
                               trainable=trainable)(face_conv_f3)
  face_flatten_f4 = layers.Flatten()(face_conv_f4)

  face_fc1 = layers.Dense(128, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(face_flatten_f4)
  face_fc2 = layers.Dense(64, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(face_fc1)

  # Face grid.
  grid_flat = layers.Flatten()(grid_floats)
  grid_fc1 = layers.Dense(256, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(grid_flat)
  grid_fc2 = layers.Dense(128, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(grid_fc1)

  # Concat everything and put through a final FF layer.
  all_concat = layers.Concatenate()([fc_e1, face_fc2, grid_fc2])
  all_fc1 = layers.Dense(128, activation="relu",
                         kernel_regularizer=l2_reg,
                         trainable=trainable)(all_concat)
  all_fc2 = layers.Dense(2, kernel_regularizer=l2_reg)(all_fc1)

  # Build the model.
  model = Model(inputs=[left_eye_input, right_eye_input, face_input,
                        grid_input],
                outputs=all_fc2)
  model.summary()

  return model
