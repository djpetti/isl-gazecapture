import numpy as np

from face_tracking import landmark_detection as ld
from face_tracking import misc

import cv2

import config


class EyeCropper:
  """ Handles croping the eye from a series of images. """

  def __init__(self):
    # Landmark detector to use henceforth.
    self.__detector = ld.LandmarkDetection()
    self.__pose = ld.PoseEstimation()

    # The last detection flag.
    self.__detect_flag = 1
    # The last detected landmark points.
    self.__points = None

    self.__image_shape = None

  def __crop_eye(self, eye_corner_l, eye_corner_r, image):
    """ Crops an eye image.
    Args:
      eye_corner_l: The left corner point of the eye.
      eye_corner_r: The right corner point of the eye.
      image: The raw image to crop the eye from.
    Returns:
      The eye crop image, scaled to 224x224. """
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

    # Convert to x, y, w, h form.
    eye_w = eye_corner_l[0] - eye_corner_r[0]
    # We want the box to be square, so make the height the same as the width.
    eye_h = eye_w

    # Also, shift the y coordinate up, so it's centered vertically.
    eye_bbox = [eye_corner_r[0], eye_corner_r[1] - eye_h / 2, eye_w, eye_h]
    # Increase the margin around the eye a little.
    eye_bbox = increase_margin(eye_bbox, 1.5)

    # Crop the eye.
    eye_crop = image[eye_bbox[0]:(eye_bbox[0] + eye_bbox[2]),
                     eye_bbox[1]:(eye_bbox[1] + eye_bbox[3])]
    # Resize.
    eye_crop = cv2.resize(eye_crop, (224, 224))

    return eye_crop

  def __crop_eyes(self, image, pts):
    """ Crops both eyes using the landmark points.
    Args:
      image: The image to crop.
      pts: The landmark points for that image.
    Returns:
      The left and right eye crops, scaled to 224x224. """
    left_crop = self.__crop_eye(pts[29], pts[26], image)
    right_crop = self.__crop_eye(pts[23], pts[20], image)

    return (left_crop, right_crop)

  def __get_face_box(self, points):
    """ Quick-and-dirty face bbox estimation based on detected points.
    Args:
      points: The detected facial landmark points. """
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

    return ((low_x, low_y), (high_x, high_y))

  def crop_image(self, image):
    """ Crops a single image.
    Args:
      image: The image to crop.
    Returns:
      The left eye, right eye, and face cropped from the image and rescaled to
      224x224, or None if it failed to crop them. """
    # Flip it to be compatible with other data.
    image = np.fliplr(image)
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

    if confidence < config.MIN_CONFIDENCE:
      # Not a good detection.
      return (None, None, None)

    # Crop the eyes.
    left_eye, right_eye = self.__crop_eyes(image, self.__points)

    # Crop the face.
    p1, p2 = self.__get_face_box(self.__points)
    face = image[p1[0]:p2[0], p1[1]:p2[1]]

    return (left_eye, right_eye, face)

  def estimate_pose(self):
    """ Returns the head pose estimate for the last image it cropped.
    Returns:
      A matrix with pitch, yaw, and roll. """
    return self.__pose.weakIterative_Occlusion(self.__points)

  def face_grid(self):
    """ Constructs the face grid input for the last image it cropped.
    Returns:
      A 25x25 matrix, where 1s represent the location of the face, and 0s
      represent the backgroud. """
    point1, point2 = self.__get_face_box(self.__points)
    p1_x, p1_y = point1
    p2_x, p2_y = point2

    # Scale to the image shape.
    image_y, image_x, _ = self.__image_shape
    p1_x = float(p1_x) / image_x * 25.0
    p2_x = float(p2_x) / image_x * 25.0
    p1_y = float(p1_y) / image_y * 25.0
    p2_y = float(p2_y) / image_y * 25.0

    p1_x = int(p1_x)
    p2_x = int(p2_x)
    p1_y = int(p1_y)
    p2_y = int(p2_y)

    # Create the interior image.
    face_box = np.ones((p2_y - p1_y, p2_x - p1_x))
    # Create the background.
    frame = np.zeros((25, 25))
    # Superimpose it.
    frame[p1_y:p2_y, p1_x:p2_x] = face_box

    return frame
