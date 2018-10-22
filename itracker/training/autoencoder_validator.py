import cPickle as pickle
import logging

import keras.backend as K
import keras.losses as losses
import numpy as np

import tensorflow as tf

import metrics
import validator


logger = logging.getLogger(__name__)


class Validator(validator.Validator):
  """ A special validator class for the autoencoder. """

  def _build_analysis_graph(self):
    # Separate data tensors.
    leye, reye, face, mask, session_num, pose = self.__data_tensors
    # TODO (danielp): Keras doesn't like it when I reshape the mask here, even
    # though it forces me to do it in the normal Validator. I think it's because
    # I don't use the mask in the autoencoder. Possible bug in Keras? Should
    # investigate.

    # Run the model.
    decoding, predicted_gaze, encoding = self._model([leye, reye, face, mask])

    # Compute the gaze error.
    self.__gaze_error = metrics.distance_metric(self.__labels["dots"],
                                                predicted_gaze)
    # Compute the decoding error.
    self.__decode_error = losses.mean_squared_error(self.__labels["decode"],
                                                    decoding)
    # Save the encoding, session number, and pose.
    self.__encoding = encoding
    self.__pose = pose
    self.__session_num = session_num

  def validate(self, num_batches):
    # Get the underlying TensorFlow session.
    session = K.tensorflow_backend.get_session()

    # Create a coordinator and run queues.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    total_gaze_error = []
    total_decode_error = []
    total_encoding = []
    total_pose = []
    total_session_num = []

    percentage = 0.0
    for i in range(0, num_batches):
      # Run the session to extract the values we need. make sure we put Keras in
      # testing mode.
      gaze_error, decode_error, encoding, pose, session_num = \
          session.run([self.__gaze_error, self.__decode_error, self.__encoding,
                       self.__pose, self.__session_num],
                      feed_dict={K.learning_phase(): 0})

      total_gaze_error.extend(gaze_error)
      total_decode_error.extend(decode_error)
      total_encoding.extend(encoding)
      total_pose.extend(pose)
      total_session_num.extend(session_num)

      new_percentage = float(i) / num_batches * 100
      if new_percentage - percentage > 0.01:
        print "Validating. (%.2f%% complete)" % (new_percentage)
        percentage = new_percentage

    coord.request_stop()
    coord.join(threads)

    print "Saving data matrix..."

    # Create data matrix. First, we need to stack pose, since that contains
    # three rows.
    pose_stack = np.stack(total_pose, axis=1)
    # Add error rows.
    gaze_error_row = np.asarray(total_gaze_error)
    gaze_error_row = np.expand_dims(gaze_error_row, 0)
    decode_error_row = np.asarray(total_decode_error)
    # Reduce across pixels so we have a single number for each image.
    decode_error_row = np.mean(decode_error_row, axis=(1, 2))
    decode_error_row = np.expand_dims(decode_error_row, 0)
    # Stack the encodings and add them.
    encoding_stack = np.stack(total_encoding, axis=1)
    # Add row for the session number.
    session_num_row = np.asarray(total_session_num)
    session_num_row = session_num_row.T

    data_matrix = np.concatenate((gaze_error_row, decode_error_row,
                                  encoding_stack, pose_stack, session_num_row),
                                 axis=0)

    # Make the variables columns.
    data_matrix = data_matrix.T

    # Save it.
    data_file = open(self._data_file, "wb")
    pickle.dump(data_matrix, data_file)
    data_file.close()
