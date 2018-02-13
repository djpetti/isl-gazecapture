#!/usr/bin/python


import argparse

import json
import os
import shutil

import cv2

import numpy as np


# Number of testing and validation sessions.
NUM_TEST_SESSIONS = 150
NUM_VAL_SESSIONS = 50


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
  crop_valid = np.asarray(crop_valid)

  return x_crop, y_crop, w_crop, h_crop, crop_valid

def extract_face_crops(images, face_data):
  """ Extract the face crop from an image.
  Args:
    images: A list of the images to process.
    face_data: The crop data for the face.
  Returns:
    A cropped version of the images, in the same order. A None value in this
    list indicates a face crop that was not valid. """
  face_x, face_y, face_w, face_h, _ = extract_crop_data(face_data)
  crops = []

  for i in range(0, len(images)):
    if images[i] is None:
      # Frame is invalid.
      crops.append(None)
      continue

    start_x = int(face_x[i])
    end_x = start_x + int(face_w[i])
    start_y = int(face_y[i])
    end_y = start_y + int(face_h[i])

    start_x = max(0, start_x)
    end_x = min(images[i].shape[1], end_x)
    start_y = max(0, start_y)
    end_y = min(images[i].shape[0], end_y)

    # Crop the image.
    crop = images[i][start_y:end_y, start_x:end_x]
    crops.append(crop)

  return crops

def generate_names(dot_info, grid_info, face_info, left_eye_info,
                   right_eye_info, session):
  """ Generates names for a set of images that match our existing data from Zac.
  Args:
    dot_info: The loaded dot information.
    grid_info: The loaded face grid information.
    left_eye_info: The loaded left eye crop information.
    right_eye_info: The loaded right eye crop information.
    face_grid: The loaded face grid information.
    session: The name of the session.
  Returns:
    A list of generated names. Names will be None if the data is not valid. """
  # Location of the dot.
  x_cam = dot_info["XCam"]
  y_cam = dot_info["YCam"]

  # Crop coordinates and sizes.
  _, _, w_face, h_face, _ = extract_crop_data(face_info)
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

  names = []
  for i in range(0, len(x_cam)):
    # Check if the frame is valid.
    if not (grid_valid[i] and leye_valid[i] and reye_valid[i]):
      names.append(None)
      continue

    # Generate a name for the frame.
    name = "gazecap_%s_f%d_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f.jpg" % \
            (session, i, x_cam[i], y_cam[i], x_grid[i], y_grid[i], w_grid[i],
             h_grid[i], x_leye[i], y_leye[i], w_leye[i], h_leye[i], x_reye[i],
             y_reye[i], w_reye[i], h_reye[i])
    names.append(name)

  return names

def save_images(frames, names, outdir):
  """ Copies the processed images to an output directory, with the correct
  names.
  Args:
    frames: The processed frames.
    names: The generated names for the frames, in order.
    outdir: The name of the directory to copy the frames to. """
  for frame, new_name in zip(frames, names):
    if new_name is None:
      # This means that the frame is invalid.
      continue

    out_path = os.path.join(outdir, new_name)
    if not cv2.imwrite(out_path, frame):
      raise RuntimeError("Failed to write image %s." % (out_path))

def load_images(frame_dir, frame_info, names):
  """ Loads images from the frame directory.
  Args:
    frame_dir: The directory to load images from.
    frame_info: The list of frame names.
    names: The new names to give to the loaded frames.
  Returns:
    A list of the loaded frames. Frames that are None aren't valid, and weren't
    loaded. """
  images = []

  for i, frame in enumerate(frame_info):
    if names[i] is None:
      # Frame is invalid anyway, don't bother loading it.
      images.append(None)
      continue

    frame_path = os.path.join(frame_dir, frame)

    image = cv2.imread(frame_path)
    if image is None:
      raise RuntimeError("Failed to read image: %s" % (frame_path))
    images.append(image)

  return images

def process_session(session_dir, out_dir):
  """ Process a session worth of data.
  Args:
    session_dir: The directory of the session.
    out_dir: The output directory to copy the images to.
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

  # Generate image names.
  names = generate_names(dot_info, grid_info, face_info, leye_info, reye_info,
                         session_name)

  # Check if we have any valid data from this session.
  for name in names:
    if name is not None:
      break
  else:
    # No valid data, no point in continuing.
    return False

  # Load images and crop faces.
  frame_dir = os.path.join(session_dir, "frames")
  frames = load_images(frame_dir, frame_info, names)
  face_crops = extract_face_crops(frames, face_info)

  # Copy images.
  save_images(face_crops, names, out_dir)

  return True

def process_dataset(dataset_dir, output_dir, start_at=None):
  """ Processes an entire dataset, one session at a time.
  Args:
    dataset_dir: The root dataset directory.
    output_dir: Where to write the output images.
    start_at: Session to start at. """
  # Create output directory.
  if not start_at:
    if os.path.exists(output_dir):
      # Remove existing direcory if it exists.
      print "Removing existing directory '%s'." % (output_dir)
      shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Make split directories.
    os.mkdir(os.path.join(output_dir, "train"))
    os.mkdir(os.path.join(output_dir, "test"))
    os.mkdir(os.path.join(output_dir, "val"))

  num_test = 0
  num_val = 0

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

    # Calculate percentage complete.
    percent = float(i) / len(sessions) * 100
    print "(%.2f%%) Processing session %s..." % (percent, item)

    # Determine which split this belongs in.
    split_dir = None
    used_test = False
    used_val = False
    if num_test < NUM_TEST_SESSIONS:
      split_dir = os.path.join(output_dir, "test")
      used_test = True
    elif num_val < NUM_VAL_SESSIONS:
      split_dir = os.path.join(output_dir, "val")
      used_val = True
    else:
      split_dir = os.path.join(output_dir, "train")

    if process_session(item_path, split_dir):
      if used_test:
        num_test += 1
      elif used_val:
        num_val += 1

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
