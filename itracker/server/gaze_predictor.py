from multiprocessing import Process, Queue
import logging
import time

import cv2

import numpy as np

from ..common import config
from ..common.eye_cropper import EyeCropper


logger = logging.getLogger(__name__)


def _is_stale(timestamp):
  """ Checks if an image is stale.
  Args:
    timestamp: The timestamp of the image to check.
  Returns:
    True if the image is stale, false otherwise. """
  if time.time() - timestamp > config.STALE_THRESHOLD:
    logger.warning("Dropping stale image from %f at %f." % \
                   (timestamp, time.time()))
    return True

  return False


class GazePredictor(object):
  """ Handles capturing eye images, and uses the model to predict gaze. """

  def __init__(self, model_file, phone, display=False):
    """
    Args:
      model_file: The saved model to load for predictions.
      phone: The configuration data for the phone we are using.
      display: If true, it will enable a debug display that shows the image
               crops. """
    # Initialize capture and prediction processes.
    self.__prediction_process = _CnnProcess(model_file)
    self.__landmark_process = _LandmarkProcess(self.__prediction_process,
                                               phone,
                                               display=display)

  def __del__(self):
    # Make sure internal processes have terminated.
    self.__landmark_process.release()
    self.__prediction_process.release()

  def predict_gaze(self):
    """ Predicts the user's gaze based on current frames.
    Returns:
      The predicted gaze point, in cm, the sequence number of the
      corresponding frame, and the timestamp. """
    # Wait for new output from the predictor.
    return self.__prediction_process.get_output()

  def process_image(self, image, seq_num):
    """ Adds a new image to the prediction pipeline.
    Args:
      image: The image to add.
      seq_num: The sequence number of the image. """
    # Add a new timestamp for this image. This allows us to drop frames that go
    # stale.
    timestamp = time.time()

    self.__landmark_process.add_new_input(image, seq_num, timestamp)

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
    model = config.NET_ARCH()
    self.__predictor = model.build()
    self.__predictor.load_weights(self.__model_file)

    while True:
      self.__predict_once()

  def __predict_once(self):
    """ Reads an image from the input queue, processes it, and writes a
    prediction to the output queue. """
    left_eye, right_eye, face, grid, seq_num, timestamp = \
        self.__input_queue.get()
    if seq_num is None:
      # A None tuple means the end of the sequence. Propagate this through the
      # pipeline.
      self.__output_queue.put((None, None, None))
      return
    if left_eye is None:
      # If we have a sequence number but no images, we got a bad detection.
      self.__output_queue.put((None, seq_num, timestamp))
      return
    if _is_stale(timestamp):
      # The image is stale, so indicate that it is invalid.
      self.__output_queue.put((None, seq_num, timestamp))
      return

    # Convert everything to floats.
    left_eye = left_eye.astype(np.float32)
    right_eye = right_eye.astype(np.float32)
    face = face.astype(np.float32)

    # Add the batch dimension.
    left_eye = np.expand_dims(left_eye, axis=0)
    right_eye = np.expand_dims(right_eye, axis=0)
    face = np.expand_dims(face, axis=0)
    grid = np.expand_dims(grid, axis=0)

    # Generate a prediction.
    pred = self.__predictor.predict([left_eye, right_eye, face, grid],
                                    batch_size=1)
    # Remove the batch dimension, and convert to Python floats.
    pred = [float(x) for x in pred[0]]

    self.__output_queue.put((pred, seq_num, timestamp))

  def add_new_input(self, left_eye, right_eye, face, grid, seq_num, timestamp):
    """ Adds a new input to be processed. Will block.
    Args:
      left_eye: The left eye crop.
      rigth_eye: The right eye crop.
      face: The face crop.
      grid: The face grid.
      seq_num: The sequence number of the image.
      timestamp: The timestamp of the image.
    """
    self.__input_queue.put((left_eye, right_eye, face, grid, seq_num,
                            timestamp))

  def get_output(self):
    """ Gets an output from the prediction process. Will block.
    Returns:
      The predicted gaze point and the sequence number. """
    return self.__output_queue.get()

class _LandmarkProcess(object):
  """ Reads images from a queue, and runs landmark detection in a separate
  process. """

  def __init__(self, cnn_process, phone, display=False):
    """
    Args:
      phone: The configuration of the phone that we are capturing data on.
      cnn_process: The _CnnProcess to send captured images to.
      display: If true, it will enable a debugging display that shows the
               detected crops on-screen. """
    self.__cnn_process = cnn_process
    self.__display = display
    self.__phone = phone

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
    image, seq_num, timestamp = self.__input_queue.get()
    if image is None:
      # A None tuple means the end of a sequence. Propagate this through the
      # pipeline.
      self.__cnn_process.add_new_input(None, None, None, None, None, None)
      return
    if _is_stale(timestamp):
      # Image is stale. Indicate that it is invalid.
      self.__cnn_process.add_new_input(None, None, None, None, seq_num,
                                       timestamp)

    # Crop the image.
    left_eye, right_eye, face = self.__cropper.crop_image(image)
    if face is None:
      # We failed to get an image.
      logger.warning("Failed to get good detection for %d." % (seq_num))
      # Send along the sequence number.
      self.__cnn_process.add_new_input(None, None, None, None, seq_num,
                                       timestamp)
      return

    # Produce face mask.
    mask = self.__cropper.face_grid()

    if self.__display:
      # Show the debugging display.
      mask_sized = cv2.resize(mask, (224, 224))
      mask_sized = np.expand_dims(mask_sized, axis=2)
      mask_sized = np.tile(mask_sized, [1, 1, 3]) * 225.0
      mask_sized = mask_sized.astype(np.uint8)

      combined = np.concatenate((left_eye, right_eye, face, mask_sized), axis=1)
      cv2.imshow("Server Detections", combined)
      cv2.waitKey(1)

    # Send it along.
    self.__cnn_process.add_new_input(left_eye, right_eye, face, mask,
                                     seq_num, timestamp)

  def run_forever(self):
    """ Reads and crops images indefinitely. """
    # Eye cropper to use for eye detection.
    self.__cropper = EyeCropper(self.__phone)

    while True:
      self.__run_once()

  def add_new_input(self, image, seq_num, timestamp):
    """ Adds a new input to be processed. Will block.
    Args:
      image: The image to process.
      seq_num: The sequence number of the image.
      timestamp: The timestamp of the image. """
    self.__input_queue.put((image, seq_num, timestamp))
