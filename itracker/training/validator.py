import cPickle as pickle
import logging

import keras.backend as K

import numpy as np

import tensorflow as tf

from ..common import config
import metrics


logger = logging.getLogger(__name__)


class Validator(object):
  """ Handles validation and statistical analysis of a model. """

  def __init__(self, data_tensors, labels, model_save,
               out_file="validation_data.pkl"):
    """
    Args:
      data_tensors: The input tensors for the model.
      labels: The label tensor for the model.
      model_save: The path to the saved model weights.
      out_file: The file in which to save the collected data. """
    self.__data_tensors = data_tensors
    self.__labels = labels
    self.__save_path = model_save
    self.__data_file = out_file

    # Build the model.
    self.__build_model()
    # Build the stuff we need to perform statistical analysis.
    self.__build_analysis_graph()

  def __build_model(self):
    """ Builds the model and loads the model weights. It also modifies
    self.__labels according to the model. """
    # Create the model.
    net = config.NET_ARCH(config.FACE_SHAPE, eye_shape=config.EYE_SHAPE,
                          data_tensors=self.__data_tensors[:4])
    self.__model = net.build()

    # Prepare the label data.
    self.__labels = net.prepare_labels(self.__labels)

    logger.info("Loading pretrained model '%s'." % (self.__save_path))
    self.__model.load_weights(self.__save_path)

  def __compute_face_pos(self, masks):
    """ Computes the position of the face, given the bitmask.
    Args:
      masks: The bitmasks to process.
    Returns:
      Index of the top left corner of the face box. """
    def face_pos(mask):
      """ Determines face position in a single mask image.
      Args:
        mask: The mask image.
      Returns:
        Index of the top left corner of the face box. """
      # For the x position, first reduce each column.
      columns = tf.count_nonzero(mask, axis=[1])
      # Find an arbitrary value in the block of 1s.
      block_index = tf.argmax(columns, axis=0)
      # Chop off everything after that.
      chopped_cols = columns[:block_index]
      # Count leading zeros.
      length = tf.cast(tf.shape(chopped_cols)[0], tf.int64)
      x_pos = length - tf.count_nonzero(chopped_cols)

      # Do the same procedure for the y position.
      rows = tf.count_nonzero(mask, axis=[0])
      block_index = tf.argmax(rows, axis=0)
      chopped_rows = rows[:block_index]
      length = tf.cast(tf.shape(chopped_rows)[0], tf.int64)
      y_pos = length - tf.count_nonzero(chopped_rows)

      return tf.stack((x_pos, y_pos))

    return tf.map_fn(face_pos, masks, back_prop=False,
                     dtype=(tf.int64))

  def __build_analysis_graph(self):
    """ Builds a portion of the graph for statistical analysis. """
    # Separate data tensors.
    leye, reye, face, mask, session_num, pose = self.__data_tensors
    # Keras wants the mask input to have a defined static shape.
    mask = tf.reshape(mask, [-1, 25, 25])

    # Run the model.
    predicted_gaze = self.__model([leye, reye, face, mask])

    # Compute the error, both as the distance, and as the raw coordinate error.
    self.__coord_error = self.__labels[0] - predicted_gaze
    self.__error = metrics.distance_metric(self.__labels[0], predicted_gaze)

    # Save the head pose so we can correlate this with the error.
    self.__pose = pose
    # Save the session num so we can analyze performance across subjects.
    self.__session_num = session_num
    # Also save some attributes from the bitmask for this purpose.
    self.__face_area = tf.count_nonzero(mask, axis=[1, 2])
    self.__face_pos = self.__compute_face_pos(mask)

  def validate(self, num_batches):
    """ Performs the actual validation.
    Args:
      num_batches: How many batches to run for the validation. """
    # Get the underlying TensorFlow session.
    session = K.tensorflow_backend.get_session()

    # Create a coordinator and run queues.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)

    total_error = []
    total_coord_error = []
    total_pose = []
    total_face_area = []
    total_face_pos = []
    total_session_num = []

    percentage = 0.0
    for i in range(0, num_batches):
      # Run the session to extract the values we need. Make sure we put Keras in
      # testing mode.
      error, coord_error, pose, face_area, face_pos, session_num = \
          session.run([self.__error, self.__coord_error, self.__pose,
                       self.__face_area, self.__face_pos, self.__session_num],
                      feed_dict={K.learning_phase(): 0})

      total_error.extend(error)
      total_coord_error.extend(coord_error)
      total_pose.extend(pose)
      total_face_area.extend(face_area)
      total_face_pos.extend(face_pos)
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
    # Stack the error.
    coord_error_stack = np.stack(total_coord_error, axis=1)
    error_row = np.asarray(total_error)
    error_row = np.expand_dims(error_row, 0)
    # Add a row for the face area.
    face_area_row = np.asarray(total_face_area)
    face_area_row = np.expand_dims(face_area_row, 0)
    # Stack the face position.
    face_pos_stack = np.stack(total_face_pos, axis=1)
    # Add a row for the session number.
    session_num_row = np.asarray(total_session_num)
    session_num_row = session_num_row.T
    data_matrix = np.concatenate((error_row, coord_error_stack, pose_stack,
                                  face_area_row, face_pos_stack,
                                  session_num_row), axis=0)

    # Make the variables columns.
    data_matrix = data_matrix.T

    # Save it.
    data_file = open(self.__data_file, "wb")
    pickle.dump(data_matrix, data_file)
    data_file.close()
