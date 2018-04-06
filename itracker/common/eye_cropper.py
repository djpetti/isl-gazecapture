import logging


import numpy as np

from face_tracking import landmark_detection as ld
from face_tracking import misc

import cv2

import config


logger = logging.getLogger(__name__)

# This is the baseline FOV that we normalize camera FOVs to. In this case, it is
# taken from the camera on an iPhone 6S.
FOV_NORM_LONG = 38.0
FOV_NORM_SHORT = 28.0

class EyeCropper:
  """ Handles croping the eye from a series of images. """

  def __init__(self, phone):
    """
    Args:
      phone: The configuration of the phone that produced the data. """
    self.__phone = phone

    # Landmark detector to use henceforth.
    self.__detector = ld.LandmarkDetection()
    self.__pose = ld.PoseEstimation()

    # The last detection flag.
    self.__detect_flag = 1
    # The last detected landmark points.
    self.__points = None

    self.__image_shape = None

  def __get_eye_bbox(self, eye_corner_l, eye_corner_r, image):
    """ Gets the bounding box for an eye image.
    Args:
      eye_corner_l: The left corner point of the eye.
      eye_corner_r: The right corner point of the eye.
      image: The raw image to crop the eye from.
    Returns:
      The bounding box of the eye crop. """
    def increase_margin(bbox, increase_fraction):
      """ Increases the margin around a bounding box.
      Args:
        bbox: The bbox to increase the margin of.
        increase_fraction: The fraction by which to increase by.
      Returns:
        A new bounding box with a wider margin. """
      abs_change_x = int(bbox[2] * increase_fraction / 2.0)
      abs_change_y = int(bbox[3] * increase_fraction / 2.0)

      new_bbox = [bbox[0] - abs_change_x, bbox[1] - abs_change_y,
                  bbox[2] + abs_change_x * 2, bbox[3] + abs_change_y * 2]

      return new_bbox

    eye_corner_l = [int(x) for x in eye_corner_l]
    eye_corner_r = [int(x) for x in eye_corner_r]

    # Convert to x, y, w, h form.
    eye_w = eye_corner_l[0] - eye_corner_r[0]
    # We want the box to be square, so make the height the same as the width.
    eye_h = eye_w

    # Also, shift the y coordinate up, so it's centered vertically.
    eye_bbox = [eye_corner_r[0], eye_corner_r[1] - eye_h / 2, eye_w, eye_h]
    # Increase the margin around the eye a little.
    eye_bbox = increase_margin(eye_bbox, 1.5)

    eye_bbox[0] = max(0, eye_bbox[0])
    eye_bbox[1] = max(0, eye_bbox[1])
    if (eye_bbox[2] <= 0 or eye_bbox[3] <= 0):
      # If the height or width is zero, this is not a useful detection.
      return None

    return eye_bbox

  def __get_eye_bboxes(self, image, pts):
    """ Gets both eye bounding boxes using the landmark points.
    Args:
      image: The image to crop.
      pts: The landmark points for that image.
    Returns:
      The bounding boxes for each eye. Either can be
      None if the detections were invalid. """
    left_bbox = self.__get_eye_bbox(pts[28], pts[25], image)
    right_bbox = self.__get_eye_bbox(pts[22], pts[19], image)

    return (left_bbox, right_bbox)

  def __get_face_bbox(self, points):
    """ Quick-and-dirty face bbox estimation based on detected points.
    Args:
      points: The detected facial landmark points.
    Returns:
      A four-element tuple with the first two elements representing a corner
      point, and the second two representing the width and height. It can also
      return None if the detection was bad. """
    # These points represent the extremeties.
    left = points[0]
    right = points[9]
    top_1 = points[2]
    top_2 = points[7]
    bot = points[40]

    # Figure out extremeties.
    low_x = int(left[0])
    high_x = int(right[0])
    low_y = int(min(top_1[1], top_2[1]))
    high_y = int(bot[1])

    # Make sure coordinates are in range.
    low_x = max(0.0, low_x)
    low_y = max(0.0, low_y)

    if (high_x - low_x < 1 or high_y - low_y < 1):
      # This is just a bad detection.
      return (None, None, None, None)

    return (low_x, low_y, high_x - low_x, high_y - low_y)

  def __extract_crop(self, image, bbox):
    """ Extracts a crop from the image defined by the bounding box.
    Args:
      image: The image to extract the crop from.
      bbox: The bounding box defining the crop.
    Returns:
      The extracted crop. """
    # Crop the image.
    crop = image[bbox[1]:(bbox[1] + bbox[3]),
                 bbox[0]:(bbox[0] + bbox[2])]
    # Resize.
    crop = cv2.resize(crop, (224, 224))

    return crop

  def get_bboxes(self, image):
    """ Takes an image, and gets the bounding boxes of the eyes and face.
    Args:
      image: The image to get bouncing boxes for.
    Returns:
      The left eye, right eye, and face bounding boxes, or None if the detection
      failed. """
    self.__image_shape = image.shape

    confidence = 0
    if self.__detect_flag > 0:
      # We have to perform the base detection.
      self.__points, self.__detect_flag, confidence = \
          self.__detector.ffp_detect(image)
    else:
      # We can continue tracking.
      self.__points, self.__detect_flag, confidence = \
          self.__detector.ffp_track(image, self.__points)

    if (confidence < config.MIN_CONFIDENCE or self.__detect_flag == 2):
      # Not a good detection.
      return (None, None, None)

    logger.debug("Confidence, detection flag: %f, %d" % (confidence,
                                                         self.__detect_flag))

    # Get the eyes.
    left_eye, right_eye = self.__get_eye_bboxes(image, self.__points)
    if (left_eye is None or right_eye is None):
      # Failed to crop because of a bad detection.
      return (None, None, None)

    # Get the face.
    face = self.__get_face_bbox(self.__points)
    if face is None:
      # Failed because of a bad detection.
      return (None, None, None)

    return (left_eye, right_eye, face)


  def crop_image(self, image):
    """ Crops a single image.
    Args:
      image: The image to crop.
    Returns:
      The left eye, right eye, and face cropped from the image and rescaled to
      224x224, or None if it failed to crop them. """
    # Get the bounding boxes.
    left_bbox, right_bbox, face_bbox = self.get_bboxes(image)
    if left_bbox is None:
      # Crop failure.
      return (None, None, None)

    # Crop.
    left_eye_crop = self.__extract_crop(image, left_bbox)
    right_eye_crop = self.__extract_crop(image, right_bbox)
    face_crop = self.__extract_crop(image, face_bbox)

    return (left_eye_crop, right_eye_crop, face_crop)

  def estimate_pose(self):
    """ Returns the head pose estimate for the last image it cropped.
    Returns:
      A matrix with pitch, yaw, and roll. """
    return self.__pose.weakIterative_Occlusion(self.__points)

  def face_grid_box(self):
    """ Computes the dimensions fot the face grid of the last image it cropped.
    Returns:
      The face grid x and y positions, as well as the width and height, all as
      one tuple. """
    image_h, image_w, _ = self.__image_shape

    # Normalize image dimensions and face box for the camera FOV.
    fov_long, fov_short = self.__phone.get_camera_fov()
    scale_w = FOV_NORM_LONG / fov_long
    scale_h = FOV_NORM_SHORT / fov_short

    diff_w = image_w - image_w * scale_w
    diff_h = image_h - image_h * scale_h
    image_w -= diff_w
    image_h -= diff_h

    # The face coordinates are going to shift, since we're effectively cutting
    # part of the image off.
    face_bbox = self.__get_face_bbox(self.__points)
    face_x, face_y, face_w, face_h = [float(x) for x in face_bbox]
    face_x -= diff_w / 2
    face_y -= diff_h / 2
    face_w += diff_w
    face_h += diff_h
    print face_bbox
    print "%f, %f" % (scale_w, scale_h)
    print "(%f, %f, %f, %f)" % (face_x, face_y, face_w, face_h)

    # Convert to 25x25 grid coordinate system.
    grid_x = face_x / image_w * 25.0
    grid_y = face_y / image_h * 25.0
    grid_w = face_w / image_w * 25.0
    grid_h = face_h / image_h * 25.0

    return (int(grid_x), int(grid_y), int(grid_w), int(grid_h))

  def face_grid(self):
    """ Constructs the face grid input for the last image it cropped.
    Returns:
      A 25x25 matrix, where 1s represent the location of the face, and 0s
      represent the background. """
    x, y, w, h = self.face_grid_box()

    x = max(0, x)
    y = max(0, y)

    # Create the interior image.
    face_box = np.ones((h, w))
    # Create the background.
    frame = np.zeros((25, 25))
    # Superimpose it.
    frame[y:(y + h), x:(x + w)] = face_box

    return frame
