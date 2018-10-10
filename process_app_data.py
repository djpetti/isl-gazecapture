#!/usr/bin/python


import argparse
import json
import os
import random
import shutil

import cv2

import numpy as np

import tensorflow as tf

from data_processing import frame_randomizer, records_output, session
from itracker.common import eye_cropper, phone_config


class VideoSession(session.Session):
  """ Specialization of Session for video data. """

  # Name of the final file that we write to a frame directory to indicate that
  # extraction completed.
  _INDEX_FILE = "frames.json"
  # Single eye cropper to use for all sessions.
  _eye_cropper = None

  def __init__(self, **kwargs):
    """
    Args:
      video_file: Location of the input video file.
      dot_data: List of dot locations for each frame, in cm.
      frames_per_dot: The number of captured frames for each dot.
      phone_config: The phone configuration to use.
      skip_frames: Number of frames to skip from the beginning of each video.
      int_freatures, float_features, byte_features: Additional feature
                                                    lists. """
    # Number of frames to skip from the beginning of each video.
    self.skip_frames = kwargs.get("skip_frames", 0)

    # The video file associated with the session.
    self.video_file = kwargs.get("video_file")
    if kwargs.get("frame_files"):
      # User shouldn't specify this.
      raise ValueError("frame_files will be set automatically.")

    # Get the dot data. It should be included here, instead of with the
    # features.
    self.dot_data = kwargs.get("dot_data")
    self.frames_per_dot = kwargs.get("frames_per_dot")

    # Get number of frames in the video.
    self.__total_frames = self.__num_usable_frames()

    phone = kwargs.get("phone_config")
    if phone is None:
      raise NameError("phone_config is a required keyword argument.")
    if VideoSession._eye_cropper is None:
      # Create a new eye cropper.
      VideoSession._eye_cropper = eye_cropper.EyeCropper(phone=phone)

    super(VideoSession, self).__init__(**kwargs)

    # Set the frame directory.
    self.frame_dir = self.__generate_frame_dir()

    # Precompute needed information.
    self.__prepare_data()

  def __num_usable_frames(self):
    """ Gets the total number of usable frames in the video.
    Returns:
      The number of total frames that we should process from the video. """
    # Get the total number of frames in the video.
    cap = cv2.VideoCapture(self.video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_dots = self.dot_data.shape[0]
    expected_frames = self.frames_per_dot * num_dots

    if num_frames > expected_frames:
      print "WARNING: Have extraneous frames in %s." % (self.video_file)
    elif num_frames < expected_frames:
      print "WARNING: Missing frames. Final dots will be ignored."
      # Don't process dots that we don't have the full number of frames for.
      num_dots = num_frames / self.frames_per_dot

    return self.frames_per_dot * num_dots

  def __generate_frame_dir(self):
    """ Generates a unique frame directory name for this session. """
    # We can base this on the video file name, since we know this is unique.
    video_name = os.path.basename(self.video_file)
    video_dir = os.path.dirname(self.video_file)
    frame_dir = os.path.join(video_dir, "%s_frames" % (video_name))

    return frame_dir

  def __should_extract(self):
    """
    Returns:
      True if we need to extract the frames. """
    # We write this file after successfully completing extraction.
    last_file = os.path.join(self.frame_dir, VideoSession._INDEX_FILE)

    if (os.path.exists(last_file)):
       # We've extracted the frames.
       return False

    # We have not extracted the frames.
    return True

  def __extract_frames(self):
    """ Pre-extracts the frames from the video file. """
    if os.path.exists(self.frame_dir):
      # Remove old stuff.
      shutil.rmtree(self.frame_dir)
    os.mkdir(self.frame_dir)

    # Open the video file.
    video_frames = cv2.VideoCapture(self.video_file)
    frame_names = []

    for i in range(0, self.__total_frames):
      # Get the next frame.
      _, frame = video_frames.read()
      if frame is None:
        raise ValueError("Video '%s' ended prematurely." % (self.__video_file))

      if i % self.frames_per_dot < self.skip_frames:
        # We want to skip frames at the beginning.
        continue

      # Write the frame.
      frame_name = "frame%d.jpg" % i
      frame_path = os.path.join(self.frame_dir, frame_name)
      if not cv2.imwrite(frame_path, frame):
        raise ValueError("Failed to save image %d." % (i,))
      frame_names.append(frame_name)

    # Write the index file containing all frames in order.
    final_path = os.path.join(self.frame_dir, VideoSession._INDEX_FILE)
    final_file = open(final_path, "w")
    json.dump(frame_names, final_file)
    final_file.close()

  def __get_face_grid(self):
    """ Computes the face grid fractional values for the last image it cropped.
    Returns:
      The fractional x, y, w, and h of the face grid. """
    # Get standard bounding box.
    grid_x, grid_y, grid_w, grid_h = self._eye_cropper.face_grid_box()

    # We have to do some numerical gymnastics here, since the training code
    # expects the grid to be 1-indexed, like in the gazecapture dataset.
    grid_x = (grid_x + 1) / 25.0
    grid_y = (grid_y + 1) / 25.0
    grid_w = (grid_w + 1) / 25.0
    grid_h = (grid_h + 1) / 25.0

    return (grid_x, grid_y, grid_w, grid_h)

  def __bbox_to_fractions(self, bbox, frame_h, frame_w):
    """ Converts a bounding box from pixels to frame fractions.
    Args:
      bbox: The bounding box, in pixels.
      frame_h: The frame height, in pixels.
      frame_w: The frame width, in pixels.
    Returns:
      The same bounding box, in frame fractions. """
    frame_w = float(frame_w)
    frame_h = float(frame_h)
    x, y, w, h = bbox

    x /= frame_w
    y /= frame_h
    w /= frame_w
    h /= frame_h

    return (x, y, w, h)

  def __detect_image(self, image):
    """ Detects bounding boxes for a single image.
    Args:
      image: The raw image that we want to process.
    Returns:
      The face bbox coordinates, the left eye bbox coordinates, the right eye bbox
      coordinates, and the face grid fractional coordinates, or a tuple of None if
      it failed to crop the image. """
    # Generate new detections.
    leye_bbox, reye_bbox, face_bbox = self._eye_cropper.get_bboxes(image)
    if face_bbox is None:
      # Failed to crop the image.
      print "WARNING: Failed to crop image."
      return (None, None, None, None)

    face_grid = self.__get_face_grid()
    leye_bbox, reye_bbox = convert_to_face_coords(list(face_bbox),
                                                  list(leye_bbox),
                                                  list(reye_bbox))

    # Convert to frame fractions.
    _, _, face_w, face_h = face_bbox

    leye_bbox = self.__bbox_to_fractions(leye_bbox, face_w, face_h)
    reye_bbox = self.__bbox_to_fractions(reye_bbox, face_w, face_h)

    return (face_bbox, leye_bbox, reye_bbox, face_grid)

  def __compute_bboxes(self):
    """ Computes the bounding boxes for all the frames. It adds features for the
    left and right eye bounding boxes. If the detection fails, the frame will be
    marked as invalid.
    Returns:
      The face bounding boxes. """
    leye_bboxes = []
    reye_bboxes = []
    face_bboxes = []
    grids = []

    percent = 0.0
    for i, frame_file in enumerate(self.frame_files):
      frame_path = os.path.join(self.frame_dir, frame_file)

      # Load the frame.
      frame = cv2.imread(frame_path)
      if frame is None:
        raise RuntimeError("Failed to read image: %s" % (frame_path))

      # Extract the bounding boxes.
      face, leye, reye, grid = self.__detect_image(frame)
      if face == None:
        # Landmark detection failed. Mark as invalid.
        self.valid[i] = 0
        leye = reye = face = grid = [0, 0, 0, 0]

      leye_bboxes.append(leye)
      reye_bboxes.append(reye)
      face_bboxes.append(face)
      grids.append(grid)

      new_percent = float(i) / len(self.frame_files) * 100.0
      if new_percent - percent > 0.01:
        percent = new_percent
        print "Detecting landmarks. (%.2f%% complete)" % (percent)

    # Add the features.
    leye_bboxes = np.stack(leye_bboxes)
    reye_bboxes = np.stack(reye_bboxes)
    grids = np.stack(grids)

    self.float_features.extend([leye_bboxes, reye_bboxes, grids])

    return face_bboxes

  def __generate_face_size(self):
    """ Generates and adds the face size feature from the face bounding box. """
    face_sizes = []
    for bbox in self.face_bboxes:
      face_size = bbox[:-2]
      face_sizes.append(face_size)

    # Add the feature.
    face_sizes = np.stack(face_sizes)
    self.float_features.append(face_sizes)

  def __add_dot_data(self):
    """ Adds a feature for the dot data. """
    # For the feature, we'll need to expand the dots so there's one entry per
    # frame.
    dots_expanded = []
    for dot in self.dot_data:
      for _ in range(0, self.frames_per_dot):
        dots_expanded.append(dot)

    # Add the feature.
    dots_expanded = np.stack(dots_expanded)
    self.float_features.append(dots_expanded)

  def __prepare_data(self):
    """ Precomputes all the necessary information from the video files. """
    # Extract the individual frames if necessary.
    if self.__should_extract():
      self.__extract_frames()

    # Add features for the dot data.
    self.__add_dot_data()

    # Add all the frame files.
    frame_index_file = file(os.path.join(self.frame_dir, self._INDEX_FILE))
    self.frame_files = json.load(frame_index_file)
    frame_index_file.close()

    # Indicate that everything is valid initially.
    self.valid = [1] * len(self.frame_files)
    # Compute the bounding boxes with the landmark detector.
    self.face_bboxes = self.__compute_bboxes()
    # Generate face size feature.
    self.__generate_face_size()

class AppSaver(records_output.Saver):
  """ Saver specialization for app data. """

  def _interpret_label_features(self, bytes_features, float_features,
                                int_features):
    dots, leye, reye, grid, face_size = float_features
    session_num = int_features[0]

    features = {"dots": dots,
                "face_size": face_size,
                "leye_box": leye,
                "reye_box": reye,
                "grid_box": grid,
                "session_num": session_num}

    return features

def convert_to_face_coords(face_bbox, leye_bbox, reye_bbox):
  """ Convert the left and right eye bounding boxes to be referenced relative to
  the face crop instead of the raw image.
  Args:
    face_bbox: The face bounding box.
    leye_bbox: The raw left eye bounding box.
    reye_bbox: The raw right eye bounding box.
  Returns:
    The left and right eye bounding boxes, referenced to the face bounding box.
  """
  leye_bbox[0] -= face_bbox[0]
  leye_bbox[1] -= face_bbox[1]

  reye_bbox[0] -= face_bbox[0]
  reye_bbox[1] -= face_bbox[1]

  return (leye_bbox, reye_bbox)

def dot_to_cm(dot_x, dot_y, phone):
  """ Converts the dot coordinates in pixels to cm, assuming a coordinate system
  as described by Mou, and a landscape orientation.
  Args:
    dot_x, dot_y: The pixel locations of the dot.
    phone: The configuration for the phone that produced the data.
  Returns:
    The x and y coordinates of the dot in cm. """
  res_long, res_short = phone.get_resolution()
  size_long, size_short = phone.get_screen_cm()
  offset_long, offset_short = phone.get_camera_offset()

  # Convert to cm directly.
  dot_x = float(dot_x) / res_short * size_short
  dot_y = float(dot_y) / res_long * size_long

  # The x and y values are actually flipped because we're working in landscape
  # mode.
  tmp = dot_x
  dot_x = dot_y
  dot_y = tmp

  # Account for the camera positioning.
  dot_x += offset_long
  dot_y += offset_short

  return (dot_x, dot_y)

def read_dot_data(data_file, phone):
  """ Reads the dot data for each image.
  Args:
    data_file: The data file for the session.
    phone: The configuration for the phone that produced the data.
  Returns:
    The number of frames for each dot, and an array of each dot position in cm, in
    order. """
  # Read the data file.
  dot_data = file(data_file).read()
  dot_data = dot_data.split("\n")

  # Extract the first item, which is the frames/dot.
  frames_per_dot = int(dot_data[0])
  # Remove trailing newlines.
  if dot_data[-1] == "":
    dot_data.pop()

  # Convert dots to actual numbers.
  dot_converted = []
  for pair in dot_data[1:]:
    x, y = pair.split()
    x, y = dot_to_cm(int(x), int(y), phone)
    dot_converted.append((x, y))

  # Numpyfy.
  dot_converted = np.stack(dot_converted)

  return (frames_per_dot, dot_converted)

def save_images(named_images, out_dir):
  """ Saves a set of images to an output directory.
  Args:
    named_images: A dictionary mapping image file names to image data.
    out_dir: The directory to write the named images into. """
  for name, image in named_images.iteritems():
    # Generate image path.
    image_path = os.path.join(out_dir, name)
    # Save the image.
    if not cv2.imwrite(image_path, image):
      raise RuntimeError("Failed to write image %s" % (image_path))

def process_session(index, video_file, data_file, randomizer, phone, skip=0):
  """ Process one session worth of data.
  Args:
    index: Unique number for the session.
    video_file: Path to the session video file.
    data_file: Path to the session data file.
    randomizer: The frame randomizer to add this session to.
    phone: The configuration for the phone that produced the data.
    skip: Number of frames to skip from the beginning of each video. """
  # Load dot data.
  frames_per_dot, dot_data = read_dot_data(data_file, phone)

  # Create the session number feature.
  session_num = np.array([[index]] * dot_data.shape[0] * frames_per_dot)
  int_features = [session_num]

  # Create a new session.
  my_session = VideoSession(video_file=video_file, dot_data=dot_data,
                            frames_per_dot=frames_per_dot, phone_config=phone,
                            int_features=int_features, skip_frames=skip)
  # Add the session to the randomizer.
  randomizer.add_session(my_session)

def process_day(index, day_dir, randomizer, phone, skip=0):
  """ Processes one day's worth of data.
  Args:
    index: A unique index for the day.
    day_dir: The directory for that day's data.
    randomizer: The frame randomizer to add sessions from this day to.
    phone: The configuration for the phone that produced the data.
    skip: Number of frames to skip from beinning of video. """
  for i, session_video in enumerate(os.listdir(day_dir)):
    # We're going to look for the video files.
    if not session_video.endswith(".mp4"):
      continue

    session_video = os.path.join(day_dir, session_video)

    # Find the corresponding data file.
    session_name = os.path.splitext(session_video)[0]
    session_dat = session_name + ".dat"
    if not os.path.exists(session_dat):
      print "WARNING: Found video but no .dat file %s." % (session_dat)
      continue

    # Generate a session index from the day index.
    session_index = index * 10 + i

    # Process the session.
    process_session(session_index, session_video, session_dat, randomizer,
                    phone, skip=skip)

def process_dataset(dataset_dir, output_dir, phone, args):
  """ Processes an entire dataset, one session at a time.
  Args:
    dataset_dir: The root dataset directory.
    output_dir: Where to write the output images.
    phone: The configuration for the phone that produced the data.
    args: Additional command-line arguments. """
  # Create output directory.
  if os.path.exists(output_dir):
    # Remove existing direcory if it exists.
    print "Removing existing directory '%s'." % (output_dir)
    shutil.rmtree(output_dir)
  os.mkdir(output_dir)

  # Create writers for writing output.
  train_record = os.path.join(output_dir, "app_data_train.tfrecord")
  test_record = os.path.join(output_dir, "app_data_test.tfrecord")
  train_writer = tf.python_io.TFRecordWriter(train_record)
  test_writer = tf.python_io.TFRecordWriter(test_record)

  # Create randomizers for each split.
  train_randomizer = frame_randomizer.FrameRandomizer()
  test_randomizer = frame_randomizer.FrameRandomizer()

  # Create savers for managing output writing.
  train_saver = AppSaver(train_randomizer, "train", train_writer)
  test_saver = AppSaver(test_randomizer, "test", test_writer)

  days = os.listdir(dataset_dir)
  # Shuffle the directories so we get a unique test split, but do it
  # deterministically so we can get the same test split every time.
  days.sort()
  random.seed(42)
  random.shuffle(days)

  # Process each day one by one.
  for i, item in enumerate(days):
    item_path = os.path.join(dataset_dir, item)
    if not os.path.isdir(item_path):
      # This is some extraneous file.
      continue

    # Determine which split it belongs in.
    randomizer = train_randomizer
    if i < args.test_days:
      randomizer = test_randomizer

    # Calculate percentage complete.
    percent = float(i) / len(days) * 100
    print "(%.2f%%) Processing day %s..." % (percent, item)

    process_day(i, item_path, randomizer, phone, skip=args.skip_frames)

  # Save out the data.
  train_saver.save_all()
  test_saver.save_all()

  train_saver.close()
  test_saver.close()

def main():
  parser = argparse.ArgumentParser("Convert data collected through the app.")
  parser.add_argument("dataset_dir", help="The root dataset directory.")
  parser.add_argument("output_dir",
                      help="The directory to write output images.")
  parser.add_argument("phone_config",
                      help="The configuration file for the phone.")

  parser.add_argument("-t", "--test_days", type=int, default=1,
                      help="Number of days to use in the test set.")
  parser.add_argument("-s", "--skip_frames", type=int, default=15,
      help="Number of frames to skip at the beginning of each video.")
  args = parser.parse_args()

  # Load the configuration.
  print "Using phone configuration: %s" % (args.phone_config)
  phone = phone_config.PhoneConfig(args.phone_config)

  process_dataset(args.dataset_dir, args.output_dir, phone, args)

if __name__ == "__main__":
  main()
