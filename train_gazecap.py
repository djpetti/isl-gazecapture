#!/usr/bin/python


import logging


def _configure_logging():
  """ Configure logging handlers. """
  # Cofigure root logger.
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler("itracker_train.log")
  file_handler.setLevel(logging.DEBUG)
  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.WARNING)
  formatter = logging.Formatter("%(name)s@%(asctime)s: " + \
      "[%(levelname)s] %(message)s")

  file_handler.setFormatter(formatter)
  stream_handler.setFormatter(formatter)

  root.addHandler(file_handler)
  root.addHandler(stream_handler)

# Some modules need a logger to be configured immediately.
_configure_logging()


# This forks a lot of processes, so we want to import it as soon as possible,
# when there is as little memory as possible in use.
from rpinets.myelin import data_loader

from six.moves import cPickle as pickle
import json
import os
import sys

import keras.applications as applications
import keras.backend as K
from keras.models import Model, load_model
import keras.layers as layers
import keras.optimizers as optimizers
import keras.regularizers as regularizers
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

import cv2

import numpy as np


batch_size = 64
# How many batches to have loaded into VRAM at once.
load_batches = 5
# Shape of the input images.
image_shape = (400, 400, 3)
# Shape of the extracted patches.
patch_shape = (390, 390)
# Shape of the input to the network.
input_shape = (224, 224, 3)

# Learning rates to set.
learning_rates = [0.001, 0.0001]
# How many iterations to train for at each learning rate.
iterations = [51864, 300000]

# Learning rate hyperparameters.
momentum = 0.9

# Where to save the network.
save_file = "eye_model_finetuned.hd5"
synsets_save_file = "synsets.pkl"
# Location of the dataset files.
dataset_files = "/training_data/gazecap_myelin/dataset"
# Location of the cache files.
cache_dir = "/training_data/gazecap_myelin"

# Validation data.
valid_dataset_files = "/training_data/gazecap_myelin_val/dataset"
valid_cache_dir = "/training_data/gazecap_myelin_val"
# Fine-tuning data.
ft_dataset_files = "/training_data/gazecap_myelin/dataset"
ft_cache_dir = "/training_data/gazecap_myelin"

# L2 regularizer for weight decay.
l2_reg = regularizers.l2(0.0005)

# Configure GPU VRAM usage.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


def create_bitmask_images(bboxes):
  """ Creates the bitmask images from the bbox points.
  Args:
    bboxes: A matrix of bounding boxes, where each box is a row vector of x, y,
            width, and height.
  Returns:
    The generated bitmask image. """
  # Scale to mask size.
  bboxes *= 25
  # It's one-indexed in the dataset.
  bboxes = np.round(bboxes).astype(np.int8) - 1

  bboxes = np.clip(bboxes, 0, 25)

  # Create the background.
  frames = np.zeros((bboxes.shape[0], 25, 25))

  for i in range(0, bboxes.shape[0]):
    # Create the interior image.
    x, y, w, h = bboxes[i];
    face_box = np.ones((h, w))

    # Superimpose it correctly.
    frames[i, y:y + h, x:x + w] = face_box

  return frames

def convert_labels(labels):
  """ Convert the raw labels from the dataset into matrices that can be fed into
  the loss function.
  Args:
    labels: The labels to convert.
  Returns:
    The converted label gaze points, left eye crop points, right eye crop
    points, and face masks box points. """
  dots = []
  leye_crops = []
  reye_crops = []
  face_masks = []
  for label in labels:
    # Extract eye crop data from the filenames.
    split = label.rstrip(".jpg").split("_")
    all_data = np.asarray(split[3:17], dtype=np.float32)

    eye_crops = all_data[6:]
    # We occasionally get values that are slightly out-of-bounds.
    eye_crops = np.clip(eye_crops, 0, 1)

    # Extract dot position.
    dots.append(all_data[0:2])
    # Extract left eye crops.
    leye_crops.append(eye_crops[0:4])
    # Extract right eye crops.
    reye_crops.append(eye_crops[4:8])
    # Extract bitmask.
    face_masks.append(all_data[2:6])

  dot_stack = np.stack(dots, axis=0)
  leye_stack = np.stack(leye_crops, axis=0)
  reye_stack = np.stack(reye_crops, axis=0)
  face_stack = np.stack(face_masks, axis=0)

  return (dot_stack, leye_stack, reye_stack, face_stack)

def maybe_flip(dot_data, leye_data, reye_data, mask_data, face_data):
  """ Randomly flips images and dots as a data augmentation procedure.
  Args:
    dot_data: The ground-truth dot locations.
    leye_data: The left eye bounding boxes.
    reye_data: The right eye bounding boaxes.
    mask_data: The face mask boxes.
    face_data: The face crops.
  Returns:
    Modified arguments in the same order. """
  # Multiplicand for flipping boxes.
  batch_size = dot_data.shape[0]
  box_flip = np.ones((batch_size, 4), dtype=np.int8)
  selections = np.random.choice([-1, 1], batch_size).astype(np.int8)
  selections_mask = 1 - np.clip(selections, 0, 1)
  selections_mask = selections_mask.astype(np.uint8)
  selections_mask_exp_1 = np.expand_dims(selections_mask, 1)
  box_flip[:, 0] *= selections

  # Multiplicand for flipping dots.
  dot_flip = box_flip[:, :2]

  # Start by subtracting the frame center from all boxes, so we can flip around
  # zero.
  leye_data[:, 0] -= 0.5
  reye_data[:, 0] -= 0.5
  mask_data[:, 0] -= 0.5

  # Now, flip the boxes around zero.
  leye_data *= box_flip
  reye_data *= box_flip
  mask_data *= box_flip

  # Add the frame center back to all the boxes.
  leye_data[:, 0] += 0.5
  reye_data[:, 0] += 0.5
  mask_data[:, 0] += 0.5

  # We need to subtract the width too, so that it's truly flipped.
  leye_data[:, 0] -= leye_data[:, 2] * selections_mask
  reye_data[:, 0] -= reye_data[:, 2] * selections_mask
  mask_data[:, 0] -= mask_data[:, 2] * selections_mask

  # For the ones that were flipped, we need to switch the left and right eye
  # data.
  left_flipped = leye_data * selections_mask_exp_1
  left_not_flipped = leye_data * (1 - selections_mask_exp_1)
  right_flipped = reye_data * selections_mask_exp_1
  right_not_flipped = reye_data * (1 - selections_mask_exp_1)
  leye_data = left_not_flipped + right_flipped
  reye_data = right_not_flipped + left_flipped

  # Flip the dots.
  dot_data *= dot_flip

  # Mask for flipping the face.
  face_mask = selections.astype(np.object)
  face_mask[face_mask == 1] = None

  # Flip the faces.
  same_slice = slice(None, None, None)
  flips = [(same_slice, slice(None, None, face_mask[i]), same_slice) \
           for i in range(0, face_data.shape[0])]
  face_flipped = np.array([img[flip] for img, flip in zip(face_data, flips)])

  # The original face_data should only have the correct elements flipped.
  return (dot_data, leye_data, reye_data, mask_data, face_flipped)

def extract_eye_crops(face_crops, leye_crops, reye_crops):
  """ Extracts the eye crops from the input face crops.
  Args:
    face_crops: The input face crops.
    leye_crops: The coordinates for the left eye crops.
    reye_crops: The coordinates for the right eye crops.
  Returns:
    The extracted left and right eye crops. """
  def get_crops(face_crops, crop_pixels):
    """ Does the actual crop extraction.
    Args:
      face_crops: The face crops to extract from.
      crop_pixels: The pixel-converted crop coordinates.
    Returns:
      An array of the crops. """
    crops = []

    for i in range(0, face_crops.shape[0]):
      face_crop = face_crops[i]
      ex, ey, ew, eh = crop_pixels[i]

      cropped = face_crop[ey:ey + eh, ex:ex + ew]
      # Resize crops.
      cropped = cv2.resize(cropped, (224, 224))
      crops.append(cropped)

    return np.stack(crops, axis=0)

  # Determine the width and height of the face crops.
  _, h_face, w_face, _ = face_crops.shape

  # Make sure everything is in range.
  leye_crops = np.clip(leye_crops, 0.0, 1.0)
  reye_crops = np.clip(reye_crops, 0.0, 1.0)

  # Calculate the actual pixels at which to crop.
  face_sizes = np.array([w_face, h_face, w_face, h_face])
  leye_pixels = leye_crops * face_sizes
  reye_pixels = reye_crops * face_sizes

  leye_pixels = leye_pixels.astype(np.uint16)
  reye_pixels = reye_pixels.astype(np.uint16)

  # Do the cropping.
  leye_extracted = get_crops(face_crops, leye_pixels)
  reye_extracted = get_crops(face_crops, reye_pixels)

  return (leye_extracted, reye_extracted)

def rescale_face(face_crops):
  """ Rescales a set of face crops to 224 x 224.
  Args:
    face_crops: The face crops to rescale.
  Returns:
    The new face crops, resized to 224 x 224. """
  # Output array.
  output = np.empty((face_crops.shape[0], 224, 224, 3))

  for i, crop in enumerate(face_crops):
    output[i] = cv2.resize(crop, (224, 224))

  return output

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

def distance_metric(y_true, y_pred):
  """ Calculates the euclidean distance between the two labels and the
  predictions.
  Args:
    y_true: The true labels.
    y_pred: The predictions.
  Returns:
    The element-wise euclidean distance between the labels and the predictions.
  """
  diff = y_true - y_pred
  sqr = K.square(diff)
  total = K.sum(sqr, axis=1)
  return K.sqrt(total)

def build_network(fine_tune=False):
  """ Builds the network.
  Args:
    fine_tune: Whether we are fine-tuning the model. If so, only the last layer
               will be trainable.
  Returns:
    The built network, ready to train. """
  trainable = not fine_tune

  left_eye_input = layers.Input(shape=input_shape, name="left_eye_input")
  right_eye_input = layers.Input(shape=input_shape, name="right_eye_input")
  # The face crop gets resized on-the-fly.
  face_shape = (patch_shape[0], patch_shape[1], input_shape[2])
  face_input = layers.Input(shape=face_shape, name="face_input")
  grid_input = layers.Input(shape=(25, 25), name="grid_input")

  left_eye_floats = K.cast(left_eye_input, "float32")
  right_eye_floats = K.cast(right_eye_input, "float32")
  face_floats = K.cast(face_input, "float32")
  grid_floats = K.cast(grid_input, "float32")

  # Resize face.
  scale_layer = layers.Lambda(lambda x: tf.image.resize_images(x, (224, 224)))
  face_scaled = scale_layer(face_floats)

  # Convert everything to grayscale.
  gray_layer = layers.Lambda(lambda x: rgb_to_grayscale(x))
  left_eye_gray = gray_layer(left_eye_floats)
  right_eye_gray = gray_layer(right_eye_floats)

  # Get pretrained VGG model for use as a base.
  vgg = applications.vgg19.VGG19(include_top=False,
                                 input_tensor=face_scaled)
  vgg_out = vgg.outputs[0]

  # Freeze all layers in VGG.
  for layer in vgg.layers:
    layer.trainable = False

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
  eye_combined_drop = layers.Dropout(0.5)(eye_combined)
  fc_e1 = layers.Dense(128, activation="relu",
                       kernel_regularizer=l2_reg)(eye_combined_drop)

  # Face layers.
  face_flatten_f4 = layers.Flatten()(vgg_out)
  face_flatten_drop = layers.Dropout(0.5)(face_flatten_f4)

  face_fc1 = layers.Dense(128, activation="relu",
                          kernel_regularizer=l2_reg,
                          trainable=trainable)(face_flatten_drop)
  face_fc2 = layers.Dense(64, activation="relu",
                          kernel_regularizer=l2_reg)(face_fc1)

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
                         kernel_regularizer=l2_reg)(all_concat)
  all_fc2 = layers.Dense(2, kernel_regularizer=l2_reg)(all_fc1)

  # Build the model.
  model = Model(inputs=[left_eye_input, right_eye_input, face_input,
                        grid_input],
                outputs=all_fc2)
  model.summary()

  return model

def process_data(face_data, labels):
  """ Helper that performs all the pre-processing on the input data.
  Args:
    face_data: The raw training face crops.
    labels: The raw labels.
  Returns:
    The converted left eye crops, right eye crops, face crops, face grids, and
    ground-truth dot locations. """
  # Process raw label names.
  dot_data, leye_data, reye_data, mask_bboxes = convert_labels(labels)
  # Randomly flip some images for data augmentation.
  dot_data, leye_data, reye_data, mask_bboxes, face_data = \
      maybe_flip(dot_data, leye_data, reye_data, mask_bboxes, face_data)
  # Generate masks.
  mask_data = create_bitmask_images(mask_bboxes)
  # Extract left and right eye crops.
  leye_crops, reye_crops = extract_eye_crops(face_data, leye_data, reye_data)

  return (leye_crops, reye_crops, face_data, mask_data, dot_data)

def train_section(model, data, learning_rate, iters):
  """ Trains for a number of iterations at one learning rate.
  Args:
    model: The model to train.
    data: The data manager to use.
    learning_rate: The learning rate to train at.
    iters: Number of iterations to train for.
  Returns:
    Training loss and testing accuracy for this section. """
  print "\nTraining at %f for %d iters.\n" % (learning_rate, iters)

  # Set the learning rate.
  opt = optimizers.SGD(lr=learning_rate, momentum=momentum)
  model.compile(optimizer=opt, loss=distance_metric, metrics=[distance_metric])

  training_loss = []
  testing_acc = []

  for i in range(0, iters / load_batches):
    # Get a new chunk of training data.
    training_data, training_labels = data.get_train_set()
    leye_crops, reye_crops, face_crops, mask_data, dot_data = \
        process_data(training_data, training_labels)

    # Train the model.
    history = model.fit([leye_crops, reye_crops, face_crops, mask_data],
                        dot_data,
                        epochs=1,
              					batch_size=batch_size)

    training_loss.extend(history.history["loss"])
    logging.info("Training loss: %s" % (history.history["loss"]))

    if not i % 10:
      testing_data, testing_labels = data.get_test_set()
      leye_crops, reye_crops, face_crops, mask_data, dot_data = \
          process_data(testing_data, testing_labels)

      loss, accuracy = model.evaluate([leye_crops, reye_crops, face_crops,
                                       mask_data],
                                      dot_data,
                                      batch_size=batch_size)

      logging.info("Loss: %f, Accuracy: %f" % (loss, accuracy))
      testing_acc.append(accuracy)

      # Save the trained model.
      model.save_weights(save_file)

  return (training_loss, testing_acc)

def main(load_model=None):
  """
  Args:
    load_model: A pretrained model to load, if specified. """
  model = build_network()
  if load_model:
    logging.info("Loading pretrained model '%s'." % (load_model))
    model.load_weights(load_model)

  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       cache_dir, dataset_files,
                                       patch_shape=patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)

  if os.path.exists(synsets_save_file):
    logging.info("Loading existing synsets...")
    data.load(synsets_save_file)


  training_acc = []
  training_loss = []
  testing_acc = []

  # Train at each learning rate.
  for lr, iters in zip(learning_rates, iterations):
    loss, acc = train_section(model, data, lr, iters)

    training_loss.extend(loss)
    testing_acc.extend(acc)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("gazecapture_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()

def validate(load_model, iterations):
  """ Validates the network.
  Args:
    load_model: A pretrained model to validate.
    iterations: How many iterations to validate for. """
  model = build_network()
  logging.info("Loading pretrained model '%s'." % (load_model))
  model.load_weights(load_model)

  # We don't actually train, but we need the compiled model for testing.
  opt = optimizers.SGD(lr=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss=distance_metric, metrics=[distance_metric])

  data = data_loader.SequentialDataManagerLoader(batch_size, load_batches,
                                                 image_shape, valid_cache_dir,
                                                 valid_dataset_files,
                                                 patch_shape=patch_shape,
                                                 pca_stddev=50,
                                                 patch_flip=False,
                                                 raw_labels=True)

  testing_acc = []

  # Train at each learning rate.
  for i in range(0, iterations):
    testing_data, testing_labels = data.get_test_set()
    leye_crops, reye_crops, face_crops, mask_data, dot_data = \
        process_data(testing_data, testing_labels)

    loss, accuracy = model.evaluate([leye_crops, reye_crops, face_crops,
                                      mask_data],
                                    dot_data,
                                    batch_size=batch_size)

    logging.info("Loss: %f, Accuracy: %f" % (loss, accuracy))
    testing_acc.append(accuracy)

  print "Mean accuracy: %f" % (np.mean(testing_acc))

  data.exit_gracefully()

def fine_tune(load_model, ft_lrs):
  """ Fine-tunes the model.
  Args:
    load_model: The model to load for fine-tuning.
    ft_lrs: List of tuples of learning rates and iteration counts for
            fine-tuning. """
  model = build_network()
  logging.info("Loading pretrained model '%s'." % (load_model))
  model.load_weights(load_model)

  data = data_loader.DataManagerLoader(batch_size, load_batches, image_shape,
                                       ft_cache_dir, ft_dataset_files,
                                       patch_shape=patch_shape,
                                       pca_stddev=50,
                                       patch_flip=False,
                                       raw_labels=True)

  if os.path.exists(synsets_save_file):
    logging.info("Loading existing synsets...")
    data.load(synsets_save_file)


  training_acc = []
  training_loss = []
  testing_acc = []

  # Train at each learning rate.
  for lr, iters in ft_lrs:
    loss, acc = train_section(model, data, lr, iters)

    training_loss.extend(loss)
    testing_acc.extend(acc)

  data.exit_gracefully()

  print "Saving results..."
  results_file = open("gazecapture_results.json", "w")
  json.dump((training_loss, testing_acc, training_acc), results_file)
  results_file.close()

if __name__ == "__main__":
  main(load_model="eye_model_finetuned.hd5")
