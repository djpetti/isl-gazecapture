#!/usr/bin/python


import argparse
import json
import os
import random
import shutil

import cv2

import numpy as np

import tensorflow as tf


# Number of testing and validation sessions.
NUM_TEST_SESSIONS = 150
NUM_VAL_SESSIONS = 50


class FrameRandomizer(object):
  """ Class that stores and randomizes frame data. """

  class Session(object):
    """ Represents a training session. """

    def __init__(self, **kwargs):
      # The combined metadata features for the session.
      self.label_features = kwargs.get("label_features")
      # The loaded frame information for the session.
      self.frame_info = kwargs.get("frame_info")
      # Whether each image is valid or not.
      self.valid = kwargs.get("valid")
      # The bbox information for each face.
      self.face_bboxes = kwargs.get("face_bboxes")

      # The frame directory for the session.
      self.frame_dir = kwargs.get("frame_dir")

    def __shuffle_list(self, to_shuffle, indices):
      """ Shuffles a list in-place.
      Args:
        to_shuffle: The list to shuffle.
        indices: The list of which indices go where. """
      old = to_shuffle[:]
      if type(to_shuffle) == np.ndarray:
        old = np.array(to_shuffle, copy=True)
      for i, index in enumerate(indices):
        to_shuffle[i] = old[index]

    def __load_crop(self, frame, face_bbox):
      """ Loads and crops the face image.
      Args:
        frame: The name of the frame file.
        face_bbox: The face bounding box data. """
      frame_path = os.path.join(self.frame_dir, frame)

      image = cv2.imread(frame_path)
      if image is None:
        raise RuntimeError("Failed to read image: %s" % (frame_path))

      # Extract the crop.
      return extract_face_crop(image, face_bbox)

    def shuffle(self):
      """ Shuffles all examples in the session. """
      indices = range(0, len(self.valid))
      random.shuffle(indices)

      # Shuffle each list.
      self.__shuffle_list(self.label_features, indices)
      self.__shuffle_list(self.frame_info, indices)
      self.__shuffle_list(self.valid, indices)
      self.__shuffle_list(self.face_bboxes, indices)

    def get_random(self):
      """ Gets the next random example, assuming they have already been
      shuffled. It raises a ValueError if there are no more examples.
      Returns:
        The next random example, including the features, frame information, and
        extracted face crop. """
      valid = False
      features = None
      frame_info = None

      # Skip any frames that are not valid.
      while not valid:
        if len(self.label_features) == 0:
          raise ValueError("No more examples.")

        features = self.label_features.pop()
        frame_info = self.frame_info.pop()
        # Can't pop from a Numpy array, but we can look at the end.
        face_bbox = self.face_bboxes[len(self.label_features)]
        valid = self.valid[len(self.label_features)]

      # Extract the face crop.
      crop = self.__load_crop(frame_info, face_bbox)

      return (features, crop)

    def num_valid(self):
      """
      Returns:
        The number of valid images in the session. """
      num_valid = 0
      for image in self.valid:
        if image:
          num_valid += 1

      return num_valid

  def __init__(self):
    # Create dictionary of sessions.
    self.__sessions = []
    self.__total_examples = 0

    # This is a list of indices representing all sessions in the dataset in
    # random order.
    self.__random_sessions = None

  def __build_random_sessions(self):
    """ Builds the random sessions list after all sessions have been added. """
    self.__random_sessions = []

    for i, session in enumerate(self.__sessions):
      self.__random_sessions.extend([i] * session.num_valid())

    # Shuffle all of them.
    random.shuffle(self.__random_sessions)

  def add_session_data(self, frame_dir, crop_info, label_features,
                       frame_info, valid):
    """ Add data for one session.
    Args:
      frame_dir: The directory containing the raw frames.
      crop_info: The face crop information.
      label_features: The features created for the image metadata.
      frame_info: The frame filename information.
      valid: Whether the corresponding images are valid or not. """
    session = self.Session()
    session.frame_dir = frame_dir
    face_bboxes = extract_crop_data(crop_info)
    face_bboxes = np.stack(face_bboxes, axis=1)
    session.face_bboxes = face_bboxes
    session.label_features = label_features
    session.frame_info = frame_info
    session.valid = valid

    self.__total_examples += session.num_valid()

    # Pre-shuffle all examples in the session.
    session.shuffle()

    self.__sessions.append(session)

  def get_random_example(self):
    """ Draws a random example from the session pool. It raises a ValueError if
    there is no more data left.
    Returns:
      The next random example, including the features and extracted face crop. """
    if len(self.__sessions) == 0:
      # No more data.
      raise ValueError("Session pool has no more data.")

    if not self.__random_sessions:
      # Build the session pick list.
      self.__build_random_sessions()

    # First, pick a random session.
    session_key = self.__random_sessions.pop()
    session = self.__sessions[session_key]

    # Now, pick a random example from within that session.
    features, crop = session.get_random()

    return (features, crop)

  def get_num_examples(self):
    """
    Returns:
      The total number of examples. """
    return self.__total_examples

def _int64_feature(value):
	""" Converts a list to an int64 feature.
	Args:
		value: The list to convert.
	Returns:
	  The corresponding feature. """
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """ Converts a list to a uint8 feature.
  Args:
    value: The list to convert.
  Returns:
    The corresponding feature. """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """ Converts a list to a float32 feature.
  Args:
    value: The list to convert.
  Returns:
    The corresponding feature. """
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def extract_crop_data(crop_info):
  """ Extracts the crop bounding data from the raw structure.
  Args:
    crop_info: The raw crop data structure.
  Returns:
    The x and y coordinates of the top left corner, and width and height of
    the crop, and whether the crop is valid. """
  x_crop = crop_info["X"]
  y_crop = crop_info["Y"]
  w_crop = crop_info["W"]
  h_crop = crop_info["H"]
  crop_valid = crop_info["IsValid"]

  x_crop = np.asarray(x_crop, dtype=np.float32)
  y_crop = np.asarray(y_crop, dtype=np.float32)
  w_crop = np.asarray(w_crop, dtype=np.float32)
  h_crop = np.asarray(h_crop, dtype=np.float32)
  crop_valid = np.asarray(crop_valid, dtype=np.float32)

  return x_crop, y_crop, w_crop, h_crop, crop_valid

def extract_face_crop(image, face_data):
  """ Extract the face crop from an image.
  Args:
    image: The image to process.
    face_data: The crop data for this image.
  Returns:
    A cropped version of the image. A None value in this
    list indicates a face crop that was not valid. """
  face_x, face_y, face_w, face_h, _ = face_data

  start_x = int(face_x)
  end_x = start_x + int(face_w)
  start_y = int(face_y)
  end_y = start_y + int(face_h)

  start_x = max(0, start_x)
  end_x = min(image.shape[1], end_x)
  start_y = max(0, start_y)
  end_y = min(image.shape[0], end_y)

  # Crop the image.
  crop = image[start_y:end_y, start_x:end_x]

  # Resize the crop.
  crop = cv2.resize(crop, (400, 400))

  return crop

def generate_label_features(dot_info, grid_info, face_info, left_eye_info,
                            right_eye_info):
  """ Generates label features for a set of data.
  Args:
    dot_info: The loaded dot information.
    grid_info: The loaded face grid information.
    left_eye_info: The loaded left eye crop information.
    right_eye_info: The loaded right eye crop information.
    face_grid: The loaded face grid information.
  Returns:
    Generated list of features, in this order: dots, face size, left eye, right eye,
    grid. It also returns a list indicating which items are valid. """
  # Location of the dot.
  x_cam = np.asarray(dot_info["XCam"], dtype=np.float32)
  y_cam = np.asarray(dot_info["YCam"], dtype=np.float32)

  # Crop coordinates and sizes.
  _, _, w_face, h_face, face_valid = extract_crop_data(face_info)
  x_leye, y_leye, w_leye, h_leye, leye_valid = extract_crop_data(left_eye_info)
  x_reye, y_reye, w_reye, h_reye, reye_valid = extract_crop_data(right_eye_info)
  # Face grid coordinates and sizes.
  x_grid, y_grid, w_grid, h_grid, grid_valid = extract_crop_data(grid_info)

  # Coerce face sizes to not have zeros for the invalid images so that division
  # works.
  w_face = np.clip(w_face, 1, None)
  h_face = np.clip(h_face, 1, None)

  # Convert everything to frame fractions.
  x_leye /= w_face
  y_leye /= h_face
  w_leye /= w_face
  h_leye /= h_face

  x_reye /= w_face
  y_reye /= h_face
  w_reye /= w_face
  h_reye /= h_face

  x_grid /= 25.0
  y_grid /= 25.0
  w_grid /= 25.0
  h_grid /= 25.0

  # Fuse arrays.
  dots = np.stack([x_cam, y_cam], axis=1)
  face_size = np.stack([w_face, h_face], axis=1)
  leye_boxes = np.stack([x_leye, y_leye, w_leye, h_leye], axis=1)
  reye_boxes = np.stack([x_reye, y_reye, w_reye, h_reye], axis=1)
  grid_boxes = np.stack([x_grid, y_grid, w_grid, h_grid], axis=1)

  # Create features.
  features = []
  for i in range(0, dots.shape[0]):
    dots_feature = _float_feature(list(dots[i]))
    face_size_feature = _float_feature(list(face_size[i]))
    leye_box_feature = _float_feature(list(leye_boxes[i]))
    reye_box_feature = _float_feature(list(reye_boxes[i]))
    grid_box_feature = _float_feature(list(grid_boxes[i]))

    features.append((dots_feature, face_size_feature, leye_box_feature,
                     reye_box_feature, grid_box_feature))

  # Generate valid array.
  valid = np.logical_and(np.logical_and(face_valid, grid_valid),
                         np.logical_and(leye_valid, reye_valid))

  return features, valid

def save_images(randomizer, split, writer):
  """ Copies the processed images to an output directory, with the correct
  names.
  Args:
    randomizer: The FrameRandomizer containing our data.
    split: The name of the split.
    writer: The records writer to write the data with. """
  i = 0
  last_percentage = 0.0
  while True:
    try:
      features, frame = randomizer.get_random_example()
    except ValueError:
      # No more frames to read.
      break

    # Calculate percentage complete.
    percent = float(i) / randomizer.get_num_examples() * 100
    if percent - last_percentage > 0.01:
      print "Processing %s split. (%.2f%% done)" % (split, percent)
      last_percentage = percent

    # Compress and serialize the image.
    ret, encoded = cv2.imencode(".jpg", frame)
    if not ret:
      print "WARNING: Encoding frame failed."
      continue
    image_feature = _bytes_feature([tf.compat.as_bytes(encoded.tostring())])

    # Create the combined feature.
    dots, face_size, leye_box, reye_box, grid_box = features
    combined_feature = {"%s/dots" % (split): dots,
                        "%s/face_size" % (split): face_size,
                        "%s/leye_box" % (split): leye_box,
                        "%s/reye_box" % (split): reye_box,
                        "%s/grid_box" % (split): grid_box,
                        "%s/image" % (split): image_feature}
    example = \
        tf.train.Example(features=tf.train.Features(feature=combined_feature))

    # Write it out.
    writer.write(example.SerializeToString())

    i += 1

def process_session(session_dir, randomizer):
  """ Process a session worth of data.
  Args:
    session_dir: The directory of the session.
    writer: The FrameRandomizer to randomize data with.
  Returns:
    True if it saved some valid data, false if there was no valid data. """
  session_name = session_dir.split("/")[-1]

  # Load all the relevant metadata.
  leye_file = file(os.path.join(session_dir, "appleLeftEye.json"))
  leye_info = json.load(leye_file)
  leye_file.close()

  reye_file = file(os.path.join(session_dir, "appleRightEye.json"))
  reye_info = json.load(reye_file)
  reye_file.close()

  face_file = file(os.path.join(session_dir, "appleFace.json"))
  face_info = json.load(face_file)
  face_file.close()

  dot_file = file(os.path.join(session_dir, "dotInfo.json"))
  dot_info = json.load(dot_file)
  dot_file.close()

  grid_file = file(os.path.join(session_dir, "faceGrid.json"))
  grid_info = json.load(grid_file)
  grid_file.close()

  frame_file = file(os.path.join(session_dir, "frames.json"))
  frame_info = json.load(frame_file)
  frame_file.close()

  # Generate label features.
  label_features, valid = generate_label_features(dot_info, grid_info,
                                                  face_info, leye_info,
                                                  reye_info)

  # Check if we have any valid data from this session.
  for image in valid:
    if image:
      break
  else:
    # No valid data, no point in continuing.
    return False

  # Add it to the randomizer.
  frame_dir = os.path.join(session_dir, "frames")
  randomizer.add_session_data(frame_dir, face_info, label_features,
                              frame_info, valid)

  return True

def process_dataset(dataset_dir, output_dir, start_at=None):
  """ Processes an entire dataset, one session at a time.
  Args:
    dataset_dir: The root dataset directory.
    output_dir: Where to write the output data.
    start_at: Session to start at. """
  # Create output directory.
  if not start_at:
    if os.path.exists(output_dir):
      # Remove existing direcory if it exists.
      print "Removing existing directory '%s'." % (output_dir)
      shutil.rmtree(output_dir)
    os.mkdir(output_dir)

  num_test = 0
  num_val = 0

  # Create writers for writing output.
  train_record = os.path.join(output_dir, "gazecapture_train.tfrecord")
  test_record = os.path.join(output_dir, "gazecapture_test.tfrecord")
  val_record = os.path.join(output_dir, "gazecapture_val.tfrecord")
  train_writer = tf.python_io.TFRecordWriter(train_record)
  test_writer = tf.python_io.TFRecordWriter(test_record)
  val_writer = tf.python_io.TFRecordWriter(val_record)

  # Create randomizers for each split.
  train_randomizer = FrameRandomizer()
  test_randomizer = FrameRandomizer()
  val_randomizer = FrameRandomizer()

  sessions = os.listdir(dataset_dir)

  # Process each session one by one.
  process = False
  for i, item in enumerate(sessions):
    item_path = os.path.join(dataset_dir, item)
    if not os.path.isdir(item_path):
      # This is some extraneous file.
      continue

    if (start_at and item == start_at):
      # We can start here.
      process = True
    if (start_at and not process):
      if num_test < NUM_TEST_SESSIONS:
        num_test += 1
      elif num_val < NUM_VAL_SESSIONS:
        num_val += 1

      continue

    # Print percentage complete.
    percent = float(i) / len(sessions) * 100
    print "Analyzing dataset. (%.2f%% done)" % (percent)

    # Determine which split this belongs in.
    writer = None
    randomizer = None
    used_test = False
    used_val = False
    if num_test < NUM_TEST_SESSIONS:
      writer = test_writer
      randomizer = test_randomizer
      used_test = True
    elif num_val < NUM_VAL_SESSIONS:
      writer = val_writer
      randomizer = val_randomizer
      used_val = True
    else:
      writer = train_writer
      randomizer = train_randomizer

    if process_session(item_path, randomizer):
      if used_test:
        num_test += 1
      elif used_val:
        num_val += 1

  # Write out everything.
  save_images(val_randomizer, "val", val_writer)
  save_images(test_randomizer, "test", test_writer)
  save_images(train_randomizer, "train", train_writer)

  train_writer.close()
  test_writer.close()
  val_writer.close()

def main():
  parser = argparse.ArgumentParser("Convert the GazeCapture dataset.")
  parser.add_argument("dataset_dir", help="The root dataset directory.")
  parser.add_argument("output_dir",
                      help="The directory to write output images.")
  parser.add_argument("-s", "--start_at", default=None,
                      help="Specify a session to start processing at.")
  args = parser.parse_args()

  process_dataset(args.dataset_dir, args.output_dir, args.start_at)

if __name__ == "__main__":
  main()
