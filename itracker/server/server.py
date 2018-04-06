from collections import deque
from multiprocessing import Process
import json
import logging
import socket
import struct

import cv2

import numpy as np

from gaze_predictor import GazePredictor


# Length of the buffer we use for reading data.
READ_BUFFER_LENGTH = 8192
# The JPEG magic at the start and end of each frame.
JPEG_MAGIC_START = bytes(b"\xFF\xD8")
JPEG_MAGIC_END = bytes(b"\xFF\xD9")
# Maximum size value before we assume that the size is invalid.
MAX_SIZE = 1000000


logger = logging.getLogger(__name__)


class Server(object):
  """ Simple socket server for the online gaze system. """

  class State(object):
    """ Keeps track of the parser state. """

    # Reading the image sequence number.
    READ_IMAGE_SEQ = "ReadImageSeq"
    # Reading the image size.
    READ_IMAGE_SIZE = "ReadImageSize"
    # Reading the first byte of the start magic.
    READ_MAGIC_START_BYTE1 = "ReadMagicStartByte1"
    # Reading the second byte of the start magic.
    READ_MAGIC_START_BYTE2 = "ReadMagicStartByte2"
    # Reading the first byte of the end magic.
    READ_MAGIC_END_BYTE1 = "ReadMagicEndByte1"
    # Reading the second byte of the end magic.
    READ_MAGIC_END_BYTE2 = "ReadMagicEndByte2"

  def __init__(self, port):
    """
    Args:
      port: The port to listen on. """
    self.__init_buffers()

    # A list of complete frames that we have received.
    self.__received_frames = deque()

    self.__listen(port)

  def __init_buffers(self):
    """ Initializes the parser buffers and state. """
    # Contains partial frame data.
    self.__current_frame = bytearray([])
    # Buffer for data read directly from the socket.
    self.__read_buffer = bytearray(b"\x00" * READ_BUFFER_LENGTH)
    # Current state of the parser.
    self.__state = self.State.READ_MAGIC_START_BYTE1

    # The size of the current image we're reading.
    self.__image_size = bytearray(b"\x00" * 4)
    # Remaining bytes of the current image that we have to read.
    self.__size_remaining = -1
    # Current byte we are reading for the image size.
    self.__image_size_index = 0

  def __listen(self, port):
    """ Builds the socket and starts listening for connections.
    Args:
      port: The port to listen on. """
    self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.__sock.bind(("", port))

    # We only want to accept one connection at a time.
    self.__sock.listen(1)

    logger.info("Now listening on port %d." % (port))

  def __extract_frame(self, sequence_num):
    """ Extracts the compressed frame stored in __current_frame,
    and adds it to the __received_frames list. It also clears the
    current_frame array.
    Args:
      sequence_num: The sequence number of the new frame.
    Returns:
      False if it tries to decode an invalid frame, true otherwise. """
    image = cv2.imdecode(np.asarray(self.__current_frame), cv2.IMREAD_COLOR)
    self.__current_frame = bytearray([])

    if image is None:
      # Failed to decode the image.
      logger.warning("Failed to read frame.")
      # Send invalid response.
      self.send_response(None, sequence_num)
      return False;
    else:
      logger.info("Got new frame.")
      self.__received_frames.appendleft((image, sequence_num))

      return True

  def __reset_state_machine(self):
    """ Resets the image reading state machine. """
    self.__current_frame = bytearray([])
    self.__size_remaining = -1
    self.__state = self.State.READ_MAGIC_START_BYTE1

  def __process_new_data(self, size):
    """ Processes a chunk of newly received data.
    Args:
      size: The size of the new chunk of data. """
    # Index of the first byte of the JPEG start sequence.
    jpeg_start_index = 0

    # Look for the JPEG delimiters.
    i = 0
    while i < size:
      byte = self.__read_buffer[i]

      # If we know the image size, we can skip until we get there.
      if self.__size_remaining > 0:
        in_buffer = min(size - i, self.__size_remaining)
        i += in_buffer
        self.__size_remaining -= in_buffer
        continue

      if self.__state == self.State.READ_MAGIC_START_BYTE1:
        # Look for the first byte of the start magic.
        if byte == ord(JPEG_MAGIC_START[0]):
          jpeg_start_index = i
          self.__state = self.State.READ_MAGIC_START_BYTE2

      elif self.__state == self.State.READ_MAGIC_START_BYTE2:
        # Look for the second byte of the start magic.
        if byte == ord(JPEG_MAGIC_START[1]):
          self.__state = self.State.READ_MAGIC_END_BYTE1
        else:
          # The second byte has to come right after the first.
          self.__state = self.State.READ_MAGIC_START_BYTE1

      elif self.__state == self.State.READ_MAGIC_END_BYTE1:
        # Look for the first byte of the end magic.
        if byte == ord(JPEG_MAGIC_END[0]):
          self.__state = self.State.READ_MAGIC_END_BYTE2

        elif self.__size_remaining != -1:
          # This means that our size was invalid. We're going to have to throw
          # this image away and manually search for the start of the next one.
          logger.warning("Lost image framing.")
          self.__reset_state_machine()

      elif self.__state == self.State.READ_MAGIC_END_BYTE2:
        # Look for the second byte of the end magic.
        if byte == ord(JPEG_MAGIC_END[1]):
          self.__state = self.State.READ_IMAGE_SEQ
          # Save the image data.
          self.__current_frame += self.__read_buffer[jpeg_start_index:(i + 1)]
        else:
          # The second byte has to come right after the first.
          self.__state = self.State.READ_MAGIC_END_BYTE1

      elif self.__state == self.State.READ_IMAGE_SEQ:
        # Read the sequence number of the image.
        self.__state = self.State.READ_IMAGE_SIZE

        # Since we read the sequence number, we can go ahead and extract the
        # image.
        logger.info("Sequence number: %d" % (byte))
        if not self.__extract_frame(byte):
          logger.error("Got invalid image. Resetting.")
          self.__reset_state_machine()

      elif self.__state == self.State.READ_IMAGE_SIZE:
        # Read the current size byte of the next image.
        self.__image_size[self.__image_size_index] = byte
        self.__image_size_index += 1

        if self.__image_size_index == 4:
          # We've read everything.
          self.__image_size_index = 0
          self.__state = self.State.READ_MAGIC_END_BYTE1

          # Unpack as an uint32.
          self.__size_remaining = struct.unpack("I", self.__image_size)[0]
          self.__size_remaining = socket.ntohl(self.__size_remaining)
          # Subtract 2, because we want to land of the first ending byte.
          self.__size_remaining -= 2
          # Assume that the next image starts directly after this.
          jpeg_start_index = i + 1

          if self.__size_remaining > MAX_SIZE:
            logger.error("Invalid image size. (%d) Resetting." % \
                         (self.__size_remaining))
            self.__reset_state_machine()

      i += 1

    if (self.__state == self.State.READ_MAGIC_END_BYTE1 or
        self.__state == self.State.READ_MAGIC_END_BYTE2):
      # In this case, we got the start of an image, but we didn't finish reading
      # it. Copy what we have from the buffer before we clear it.
      self.__current_frame += self.__read_buffer[jpeg_start_index:size]

  def wait_for_client(self):
    """ Waits until a client connects to the server. """
    logger.info("Waiting for client connection...")

    self.__client, self.__addr = self.__sock.accept()

    logger.info("Got new connection from %s." % (str(self.__addr)))

  def read_next_frame(self):
    """ Gets and returns the next complete JPEG frame from the client.
    Returns:
      The next frame and sequence number, or a None tuple if the client
      disconnected. """
    # Read data from the socket until we have at least one new frame.
    while len(self.__received_frames) == 0:
      logger.debug("Waiting for new data...")

      try:
        bytes_read = self.__client.recv_into(self.__read_buffer)
      except socket.error:
        # Client disconnected not-nicely.
        logger.info("Client %s disconnected (forced)." % (str(self.__addr)))
        return (None, None)

      if bytes_read == 0:
        # Assume client disconnect.
        logger.info("Client %s disconnected." % (str(self.__addr)))
        self.__client.close()

        # Clear the buffers.
        self.__init_buffers()

        return (None, None)

      self.__process_new_data(bytes_read)

    return self.__received_frames.pop()

  def send_response(self, prediction, seq_num):
    """ Sends a response for a particular image prediction.
    Args:
      prediction: The prediction, as a tuple, in cm. If None, it assumes
                  the prediction is invalid.
      seq_num: The sequence number of the image that the prediction is for.
      valid: Whether the prediction is valid. """
    logger.debug("Sending prediction %s for %d." % (str(prediction), seq_num))

    # Create JSON data to send.
    response = {"SequenceNumber": seq_num, "Valid": prediction is not None}
    if prediction is not None:
      response["PredictX"] = prediction[0]
      response["PredictY"] = prediction[1]
    response = json.dumps(response)

    # Add a delimiter to the end.
    response += "\n"

    # Send the response on the socket.
    self.__client.sendall(response)


class ReceiveProcess(object):
  """ Process that handles receiving data from the server and passing it to the
  gaze predictor. """

  def __init__(self, server, gaze_predictor):
    """
    Args:
      server: The server instance to receive data on. It expects a client to
              already be connected.
      gaze_predictor: The gaze predictor to send images to. """
    self.__server = server
    self.__predictor = gaze_predictor

    # The underlying process to run.
    self.__process = Process(target=ReceiveProcess.run_forever, args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates the internal process. """
    self.__process.terminate()

  def __run_once(self):
    """ Runs one iteration of the receiver process.
    Returns:
      True if it got a new frame, false if the client disconnected. """
    frame, sequence_num = self.__server.read_next_frame()

    # Now, send it to the gaze predictor.
    self.__predictor.process_image(frame, sequence_num)
    return frame is not None

  def run_forever(self):
    """ Runs the process forever. """
    # Run iterations.
    while self.__run_once():
      pass

    logger.info("Detected client disconnect, exiting receive process.")

  def wait_for_disconnect(self):
    """ Waits until the client disconnects. """
    self.__process.join()

class SendProcess(object):
  """ Process that handles reading results from the gaze predictor and sending
  them out with the server. """
  def __init__(self, server, gaze_predictor):
    """
    Args:
      server: The server instance to send data on. It expects a client to
              already be conneted.
      gaze_predictor: The gaze predictor to read results from. """
    self.__server = server
    self.__predictor = gaze_predictor

    # The underlying process to run.
    self.__process = Process(target=SendProcess.run_forever, args=(self,))
    self.__process.start()

  def release(self):
    """ Cleans up and terminates the internal process. """
    self.__process.terminate()

  def __run_once(self):
    """ Runs one interation of the sender process.
    Returns:
      True if it sent new data, false if the sequence end was reached. """
    gaze_point, seq_num = self.__predictor.predict_gaze()
    if seq_num is None:
      # A None tuple means the end of the sequence, so we'll want to join this
      # process.
      return False

    # Send it to the client.
    try:
      self.__server.send_response(gaze_point, seq_num)
      print gaze_point
    except socket.error:
      # The client has diconnected. We're going to wait to exit though until we
      # see the end of the sequence propagated through the pipeline.
      logger.debug("Send socket disconnected, waiting for sequence end.")

    return True

  def run_forever(self):
    """ Runs the process forever. """
    # Run iterations.
    while self.__run_once():
      pass

    logger.info("Detected client disconnect, exiting send process.")

  def wait_for_disconnect(self):
    """ Waits until the client disconnects. """
    self.__process.join()
