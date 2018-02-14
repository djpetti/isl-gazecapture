from collections import deque
import logging
import socket

import cv2

import numpy as np


# Length of the buffer we use for reading data.
READ_BUFFER_LENGTH = 8192
# The JPEG magic at the start and end of each frame.
JPEG_MAGIC_START = bytes(b"\xFF\xD8")
JPEG_MAGIC_END = bytes(b"\xFF\xD9")


logger = logging.getLogger(__name__)


class Server(object):
  """ Simple socket server for the online gaze system. """

  class State(object):
    """ Keeps track of the parser state. """

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
    self.__read_buffer = bytearray([0] * READ_BUFFER_LENGTH)
    # Current state of the parser.
    self.__state = self.State.READ_MAGIC_START_BYTE1

  def __listen(self, port):
    """ Builds the socket and starts listening for connections.
    Args:
      port: The port to listen on. """
    self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.__sock.bind(("", port))

    # We only want to accept one connection at a time.
    self.__sock.listen(1)

    logger.info("Now listening on port %d." % (port))

  def __extract_frame(self):
    """ Extracts the compressed frame stored in __current_frame,
    and adds it to the __received_frames list. It also clears the
    __current_frame array. """
    image = cv2.imdecode(np.asarray(self.__current_frame), cv2.IMREAD_COLOR)
    if image is None:
      # Failed to decode the image.
      logger.warning("Failed to read frame.")

    logger.debug("Got new frame.")
    self.__received_frames.appendleft(image)

    self.__current_frame = bytearray([])

  def __process_new_data(self, size):
    """ Processes a chunk of newly received data.
    Args:
      size: The size of the new chunk of data. """
    # Index of the first byte of the JPEG start sequence.
    jpeg_start_index = 0

    # Look for the JPEG delimiters.
    for i in range(0, size):
      byte = self.__read_buffer[i]

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

      elif self.__state == self.State.READ_MAGIC_END_BYTE2:
        # Look for the second byte of the end magic.
        if byte == ord(JPEG_MAGIC_END[1]):
          self.__state = self.State.READ_MAGIC_START_BYTE1
          # Copy the full image data.
          self.__current_frame += self.__read_buffer[jpeg_start_index:(i + 1)]
          self.__extract_frame()

        else:
          # The second byte has to come right after the first.
          self.__state = self.State.READ_MAGIC_END_BYTE1

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
      The next frame, or None if the client disconnected. """
    # Read data from the socket until we have at least one new frame.
    while len(self.__received_frames) == 0:
      logger.debug("Waiting for new data...")
      bytes_read = self.__client.recv_into(self.__read_buffer)
      if bytes_read == 0:
        # Assume client disconnect.
        logger.info("Client %s disconnected." % (str(self.__addr)))
        self.__client.close()

        # Clear the buffers.
        self.__init_buffers()

        return None

      self.__process_new_data(bytes_read)

    return self.__received_frames.pop()
