#!/usr/bin/python


import argparse
import os
import shutil

import cv2

class BluePhone(object):
  """ Specifications for BluePhone. """

  # Phone screen dimensions in cm.
  SCREEN_LONG_CM = 11.05
  SCREEN_SHORT_CM = 6.25

  # Screen resolution, in pixels.
  RES_LONG = 1280
  RES_SHORT = 720

  # Camera positioning, in cm.
  CAMERA_LONG_OFFSET = 1.05
  CAMERA_SHORT_OFFSET = -1.85

class Nexus6P(object):
  """ Specifications for Nexus 6P. """

  # Phone screen dimensions in cm.
  SCREEN_LONG_CM = 12.55
  SCREEN_SHORT_CM = 7.10

  # Screen resolution, in pixels.
  RES_LONG = 2560
  RES_SHORT = 1440

  # Camera positioning, in cm.
  CAMERA_LONG_OFFSET = 0.95
  CAMERA_SHORT_OFFSET = -1.25

# Which phone specs to use.
PHONE = BluePhone


def load_image_and_data(session_dir, image_name):
  """ Loads an image and corresponding data for that image.
  Args:
    session_dir: The base session directory that the image is in.
    image_name: The name of the image to load data for, without the extension.
  Returns:
    The loaded image, the face bbox coordinates, the left eye bbox coordinates,
    and the right eye bbox coordinates. """
  def to_ints(coord_string):
    """ Helper function that converts a space-delimited string of numbers to a
    list of ints.
    Args:
      coord_string: The input string of numbers.
    Returns:
      The output int string. """
    ret = coord_string.split(" ")
    return [int(float(x)) for x in ret]

  def increase_margin(bbox, increase_fraction):
    """ Increases the margin around a bounding box.
    Args:
      bbox: The bbox to increase the margin of.
      increase_fraction: The fraction by which to increase by.
    Returns:
      A new bounding box with a wider margin. """
    abs_change_x = int(bbox[2] * increase_fraction / 2)
    abs_change_y = int(bbox[3] * increase_fraction / 2)

    new_bbox = [bbox[0] - abs_change_x, bbox[1] - abs_change_y,
                bbox[2] + abs_change_x * 2, bbox[3] + abs_change_y * 2]

    return new_bbox

  # Paths to the image, and the data file.
  image_path = os.path.join(session_dir, "raw", image_name + ".jpg")
  data_path = os.path.join(session_dir, "detectionResult", image_name + ".dat")
  if not os.path.exists(image_path):
    raise RuntimeError("Image path '%s' does not exist!" % (image_path))
  if not os.path.exists(data_path):
    raise RuntimeError("Data path '%s' does not exist!" % (data_path))

  # Load the raw image.
  image = cv2.imread(image_path)
  if image is None:
    raise RuntimeError("Failed to load image '%s'." % (image_path))

  # Read the data file.
  data_file = file(data_path)
  detection_data = data_file.read()
  detection_data = detection_data.split("\n")

  # The first line should give us our face bbox.
  face_bbox = to_ints(detection_data[0])

  # Extract the points for the left and right corners of each eye.
  leye_corner_l = to_ints(detection_data[29])
  leye_corner_r = to_ints(detection_data[26])
  # Convert to x, y, w, h form.
  leye_w = leye_corner_l[0] - leye_corner_r[0]
  # We want the box to be square, so make the height the same as the width.
  leye_h = leye_w
  # Also, shift the y coordinate up, so it's centered vertically.
  leye_bbox = [leye_corner_r[0], leye_corner_r[1] - leye_h / 2, leye_w, leye_h]
  # Increase the margin around the eye a little.
  leye_bbox = increase_margin(leye_bbox, 1.5)

  reye_corner_l = to_ints(detection_data[23])
  reye_corner_r = to_ints(detection_data[20])
  reye_w = reye_corner_l[0] - reye_corner_r[0]
  reye_h = reye_w
  reye_bbox = [reye_corner_r[0], reye_corner_r[1] - reye_h / 2, reye_w, reye_h]
  # Increase the margin around the eye a little.
  reye_bbox = increase_margin(reye_bbox, 1.5)

  return (image, face_bbox, leye_bbox, reye_bbox)

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

def get_face_grid(image_size, face_bbox):
  """ Computes the face grid fractional values for a face crop.
  Args:
    image_size: The shape of the raw image.
    face_bbox: The face bounding box.
  Returns:
    The fractional x, y, w, and h of the face grid. """
  image_h, image_w, _ = image_size

  grid_x = float(face_bbox[0]) / image_w
  grid_y = float(face_bbox[1]) / image_h

  grid_w = float(face_bbox[2]) / image_w
  grid_h = float(face_bbox[3]) / image_h

  # We have to do some numerical gymnastics here, since the training code
  # expects the grid to be 1-indexed, like in the gazecapture dataset.
  grid_x = ((grid_x * 25.0) + 1) / 25.0
  grid_y = ((grid_y * 25.0) + 1) / 25.0
  grid_w = ((grid_w * 25.0) + 1) / 25.0
  grid_h = ((grid_h * 25.0) + 1) / 25.0

  return (grid_x, grid_y, grid_w, grid_h)

def extract_crop(image, bbox):
  """ Extracts a crop from a raw image.
  Args:
    image: The image to extract from.
    bbox: The bounding box to extract.
  Returns:
    The extracted crop. """
  x, y, w, h = bbox
  return image[y:y + h, x:x + w]

def load_face_and_data(session_dir, image_name):
  """ Loads an image and corresponding data for that image, and extracts the
  face.
  Args:
    session_dir: The base session directory that the image is in.
    image_name: The name of the image to load data for, without the extension.
  Returns:
    The loaded face crop, the left eye bbox coordinates, the right eye bbox
    coordinates, and the face grid fractional coordinates. """
  # Load the image and data.
  image, face_bbox, leye_bbox, reye_bbox = load_image_and_data(session_dir,
                                                               image_name)

  face_grid = get_face_grid(image.shape, face_bbox)
  face_crop = extract_crop(image, face_bbox)
  leye_bbox, reye_bbox = convert_to_face_coords(face_bbox, leye_bbox, reye_bbox)

  return (face_crop, leye_bbox, reye_bbox, face_grid)

def display_crops(image, leye_bbox, reye_bbox):
  """ Displays an image, with the face and eye bounding boxes shown.
  Args:
    image: The image to display.
    leye_bbox: The left eye bounding box.
    reye_bbox: The right eye bounding box. """
  vis = image.copy()

  leye_x, leye_y, leye_w, leye_h = leye_bbox
  reye_x, reye_y, reye_w, reye_h = reye_bbox

  cv2.rectangle(vis, (leye_x, leye_y), (leye_x + leye_w, leye_y + leye_h),
                (0, 255, 0))
  cv2.rectangle(vis, (reye_x, reye_y), (reye_x + reye_w, reye_y + reye_h),
                (0, 0, 255))

  cv2.imshow("test", vis)
  cv2.waitKey(0)

def dot_to_cm(dot_x, dot_y):
  """ Converts the dot coordinates in pixels to cm, assuming a coordinate system
  as described by Mou, and a landscape orientation.
  Args:
    dot_x, dot_y: The pixel locations of the dot.
  Returns:
    The x and y coordinates of the dot in cm. """
  # Convert to cm directly.
  dot_x = float(dot_x) / PHONE.RES_SHORT * PHONE.SCREEN_SHORT_CM
  dot_y = float(dot_y) / PHONE_RES_LONG * PHONE.SCREEN_LONG_CM

  # The x and y values are actually flipped because we're working in landscape
  # mode.
  tmp = dot_x
  dot_x = dot_y
  dot_y = tmp

  # Account for the camera positioning.
  dot_x += PHONE.CAMERA_LONG_OFFSET
  dot_y += PHONE.CAMERA_SHORT_OFFSET

  return (dot_x, dot_y)

def read_dot_data(session_dir):
  """ Reads the dot data for each image.
  Args:
    session_dir: The directory where the session is located.
  Returns:
    A dictionary that maps image names to the dot position in cm. """
  # Paths to the order file and the dot file.
  order_path = os.path.join(session_dir, "leftEye", "order.dat")
  dot_path = os.path.join(session_dir, "leftEye", "XY.dat")

  order_data = file(order_path).read()
  dot_data = file(dot_path).read()

  order_data = order_data.split("\n")
  order_data = [x.rstrip(".dat") for x in order_data]
  dot_data = dot_data.split("\n")

  # Remove trailing newlines.
  if order_data[-1] == "":
    order_data.pop()
  if dot_data[-1] == "":
    dot_data.pop()

  # Convert dots to actual numbers.
  dot_converted = []
  for pair in dot_data:
    x, y = pair.split()
    x, y = dot_to_cm(int(x), int(y))
    dot_converted.append((x, y))

  # Put into a dictionary.
  ret = {}
  for i in range(0, len(order_data)):
    ret[order_data[i]] = dot_converted[i]

  return ret

def load_session(session_dir):
  """ Loads an entire session worth of data.
  Args:
    session_dir: The directory where the session data is located.
  Returns:
    A dictionary mapping the image name to a list containing the face crop,
    dot location, left eye bbox, right eye bbox, and face grid. """
  # Load the total list of images and dot coordinates.
  dot_data = read_dot_data(session_dir)

  image_data = {}
  for image_name, dot_coords in dot_data.iteritems():
    # Load face and calculate bbox data.
    face_crop, leye_bbox, reye_bbox, grid = load_face_and_data(session_dir,
                                                               image_name)
    data_list = [face_crop, dot_coords, leye_bbox, reye_bbox, grid]

    image_data[image_name] = data_list

  return image_data

def generate_names(session, image_data):
  """ Generates names for the images that incorporate the label.
  Args:
    session: The name of the session.
    image_data: The dictionary produced by load_session.
  Returns:
    A dictionary that maps generated image names to face crop images. """
  names = {}
  for i, data in enumerate(image_data.itervalues()):
    face_crop, dot_coords, leye_bbox, reye_bbox, grid = data

    dot_x, dot_y = dot_coords

    # Extract the size of the face crop.
    h_face, w_face, _ = face_crop.shape
    h_face = float(h_face)
    w_face = float(w_face)

    x_leye, y_leye, w_leye, h_leye = leye_bbox
    x_reye, y_reye, w_reye, h_reye = reye_bbox
    x_grid, y_grid, w_grid, h_grid = grid

    # Convert everything to frame fractions.
    x_leye /= w_face
    y_leye /= h_face
    w_leye /= w_face
    h_leye /= h_face

    x_reye /= w_face
    y_reye /= h_face
    w_reye /= w_face
    h_reye /= h_face

    # Generate a name for the frame.
    name = "gazecap_%s_f%d_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f_%f.jpg" % \
          (session, i, dot_x, dot_y, x_grid, y_grid, w_grid,
           h_grid, x_leye, y_leye, w_leye, h_leye, x_reye,
           y_reye, w_reye, h_reye)

    names[name] = face_crop

  return names

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

def process_session(session_dir, out_dir):
  """ Process one session worth of data.
  Args:
    session_dir: The directory of the session.
    out_dir: The output directory to copy the images to. """
  # Load the session data.
  session_data = load_session(session_dir)

  # Generate names for the images.
  session_name = session_dir.split("/")[-1]
  # Remove underscores from the session name, since we use that as a separator
  # character.
  session_name = session_name.replace("_", "")
  name_data = generate_names(session_name, session_data)

  # Write the images to the output directory.
  save_images(name_data, out_dir)

def process_dataset(dataset_dir, output_dir):
  """ Processes an entire dataset, one session at a time.
  Args:
    dataset_dir: The root dataset directory.
    output_dir: Where to write the output images. """
  # Create output directory.
  if os.path.exists(output_dir):
    # Remove existing direcory if it exists.
    print "Removing existing directory '%s'." % (output_dir)
    shutil.rmtree(output_dir)
  os.mkdir(output_dir)

  sessions = os.listdir(dataset_dir)

  # Process each session one by one.
  for i, item in enumerate(sessions):
    item_path = os.path.join(dataset_dir, item)
    if not os.path.isdir(item_path):
      # This is some extraneous file.
      continue

    # Calculate percentage complete.
    percent = float(i) / len(sessions) * 100
    print "(%.2f%%) Processing session %s..." % (percent, item)

    process_session(item_path, output_dir)

def main():
  parser = argparse.ArgumentParser("Convert Mou's dataset.")
  parser.add_argument("dataset_dir", help="The root dataset directory.")
  parser.add_argument("output_dir",
                      help="The directory to write output images.")
  args = parser.parse_args()

  process_dataset(args.dataset_dir, args.output_dir)

if __name__ == "__main__":
  main()
