from multiprocessing import Process, Queue
import logging
import time

import cv2

from keras.models import load_model

import numpy as np

from face_tracking import landmark_detection as ld
from face_tracking import misc

from eye_cropper import EyeCropper


logger = logging.getLogger(__name__)


class GazePredictor(object):
  """ Handles capturing eye images, and uses the model to predict gaze. """

  def __init__(self, model_file):
    """
    Args:
      model_file: The saved model to load for predictions. """
    # Initialize capture and prediction processes.
    self.__prediction_process = _CnnProcess(model_file)
    self.__landmark_process = _LandmarkProcess(self.__prediction_process)

  def __del__(self):
    # Make sure internal processes have terminated.
    self.__landmark_process.release()
    self.__prediction_process.release()

  def predict_gaze(self):
    """ Predicts the user's gaze based on current frames.
    Returns:
      The predicted gaze point, in cm, as well as the sequence number of the
      corresponding frame. """
    # Wait for new output from the predictor.
    return self.__prediction_process.get_output()

  def process_image(self, image, seq_num):
    """ Adds a new image to the prediction pipeline.
    Args:
      image: The image to add.
      seq_num: The sequence number of the image. """
    self.__landmark_process.add_new_input(image, seq_num)

class _CnnProcess(object):
  """ Runs the CNN prediction in a separate process on the GPU, so that it can
  be handled concurrently. """

  def __init__(self, model_file):
    """
    Args:
      model_file: The file to load the predictor model from. """
    self.__model_file = model_file

    # Create the queues.
    self.__input_queue = Queue()
    self.__output_queue = Queue()

    # Fork the predictor process.
    self.__process = Process(target=_CnnProcess.predict_forever, args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()

  def predict_forever(self):
    """ Generates predictions indefinitely. """
    # Load the model we trained.
    #self.__predictor = load_model(self.__model_file)

    while True:
      self.__predict_once()

  def __predict_once(self):
    """ Reads an image from the input queue, processes it, and writes a
    prediction to the output queue. """
    left_eye, right_eye, face, grid, seq_num = self.__input_queue.get()
    if left_eye is None:
      # A None tuple means the end of the sequence. Propagate this through the
      # pipeline.
      self.__output_queue.put((None, None))
      return

    # Convert everything to floats.
    left_eye = left_eye.astype(np.float32)
    right_eye = right_eye.astype(np.float32)
    face = face.astype(np.float32)

    # Generate a prediction.
    #pred = self.__predictor.predict([left_eye, right_eye, face, grid],
    #                                batch_size=1)
    pred = (0.0, 0.0)

    self.__output_queue.put((pred, seq_num))

  def add_new_input(self, left_eye, right_eye, face, grid, seq_num):
    """ Adds a new input to be processed. Will block.
    Args:
      left_eye: The left eye crop.
      rigth_eye: The right eye crop.
      face: The face crop.
      grid: The face grid.
      seq_num: The sequence number of the image.
    """
    self.__input_queue.put((left_eye, right_eye, face, grid, seq_num))

  def get_output(self):
    """ Gets an output from the prediction process. Will block.
    Returns:
      The predicted gaze point and the sequence number. """
    return self.__output_queue.get()

class _LandmarkProcess(object):
  """ Reads images from a queue, and runs landmark detection in a separate
  process. """

  def __init__(self, cnn_process):
    """
    Args:
      cnn_process: The _CnnProcess to send captured images to. """
    self.__cnn_process = cnn_process

    # Create the queues.
    self.__input_queue = Queue()
    self.__output_queue = Queue()

    # Fork the capture process.
    self.__process = Process(target=_LandmarkProcess.run_forever,
                             args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates internal process. """
    self.__process.terminate()

  def __run_once(self):
    """ Reads and crops a single image. It will send it to the predictor
    process when finished. """
    # Get the next input from the queue.
    image, seq_num = self.__input_queue.get()
    if image is None:
      # A None tuple means the end of a sequence. Propagate this through the
      # pipeline.
      self.__cnn_process.add_new_input(None, None, None, None, None)
      return

    # Crop the image.
    left_eye, right_eye, face = self.__cropper.crop_image(image)
    if face is None:
      # We failed to get an image.
      logger.warning("Failed to get good detection for %d." % (seq_num))
      return

    # Produce face mask.
    mask = self.__cropper.face_grid()

    # Send it along.
    self.__cnn_process.add_new_input(left_eye, right_eye, face, mask,
                                     seq_num)

  def run_forever(self):
    """ Reads and crops images indefinitely. """
    # Eye cropper to use for eye detection.
    self.__cropper = EyeCropper()

    while True:
      self.__run_once()

  def add_new_input(self, image, seq_num):
    """ Adds a new input to be processed. Will block.
    Args:
      image: The image to process.
      seq_num: The sequence number of the image. """
    self.__input_queue.put((image, seq_num))
