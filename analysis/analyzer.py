import cPickle as pickle

import matplotlib.pyplot as plt

import numpy as np


class Analyzer:
  """ Handles analysis for the validation data. """

  # Constants defining the indices of the variable columns in the data matrix.
  _ERROR_COL = 0
  _ERROR_X_COL = 1
  _ERROR_Y_COL = 2
  _HEAD_PITCH_COL = 3
  _HEAD_YAW_COL = 4
  _HEAD_ROLL_COL = 5
  _FACE_AREA_COL = 6
  _FACE_POS_Y_COL = 7
  _FACE_POS_X_COL = 8
  _SESSION_NUM_COL = 9

  def __init__(self, data_file):
    """
    Args:
      data_file: The file to load the validation data from. """
    self.__load_data(data_file)

    # Covariance matrix.
    self.__sigma = None
    # Correlation matrix.
    self.__correlation = None
    # Per-subject error sequence.
    self.__subject_errors = []

  def __load_data(self, data_file):
    """ Loads the data from a file. """
    print "Loading validation data..."
    self.__data = pickle.load(file(data_file, "rb"))

  def __cov(self):
    """ Computes the covariance matrix, if it hasn't been already.
    Returns:
      The computed covariance matrix. """
    if self.__sigma is None:
      # We need to specify that are variables are columns.
      self.__sigma = np.cov(self.__data, rowvar=False)

    return self.__sigma

  def __corr(self):
    """ Computes the correlation matrix, if it hasn't been already.
    Returns:
      The computed correlation matrix. """
    if self.__correlation is None:
      # We take the absolute value when calculating correlation because for the
      # attributes we're interested in, we expect the error to be smallest when
      # they're around zero, and larger the farther away they get.
      centered = self.__data - np.mean(self.__data, axis=0)
      self.__correlation = np.corrcoef(centered, rowvar=False)

    return self.__correlation

  def __per_subject_error(self):
    """ Separates the error out on a per-subject basis.
    Returns:
      A list of vectors, where each vector contains error data for a single
      subject. """
    if self.__subject_errors:
      # We've already generated this.
      return self.__subject_errors

    # Generate it.
    subject_col = self.__data[:, self._SESSION_NUM_COL]
    subjects = np.unique(subject_col)

    # Compute accuracies for each subject.
    subject_errors = []
    all_error = self.__data[:, self._ERROR_COL]
    for subject in subjects:
      error_data = all_error[subject_col == subject]
      self.__subject_errors.append(error_data)

    return self.__subject_errors

  def __mean_accuracy(self):
    """ Computes the mean accuracy.
    Returns:
      The mean accuracy. """
    return np.mean(self.__data[:, self._ERROR_COL])

  def __mean_pose(self):
    pitch = np.mean(self.__data[:, self._HEAD_PITCH_COL])
    yaw = np.mean(self.__data[:, self._HEAD_YAW_COL])
    roll = np.mean(self.__data[:, self._HEAD_ROLL_COL])

    return (pitch, yaw, roll)

  def __mean_face_area(self):
    """ Computes the mean face area.
    Returns:
      The mean face area. """
    return np.mean(self.__data[:, self._FACE_AREA_COL])

  def __mean_face_pos(self):
    """ Computes the mean face position.
    Returns:
      The mean y and x values of the face position. """
    pos_mu_y = np.mean(self.__data[:, self._FACE_POS_Y_COL])
    pos_mu_x = np.mean(self.__data[:, self._FACE_POS_X_COL])

    return (pos_mu_y, pos_mu_x)

  def __stddev_accuracy(self):
    """ Computes the standard deviation of the accuracy.
    Returns:
      The standard deviation of the accuracy. """
    sigma = self.__cov()
    return np.sqrt(sigma[self._ERROR_COL, self._ERROR_COL])

  def __stddev_pose(self):
    """ Computes the standard deviation of the head pose.
    Returns:
      The standard deviation of the head pose, across pitch, yaw, and roll. """
    sigma = self.__cov()

    pitch = sigma[self._HEAD_PITCH_COL, self._HEAD_PITCH_COL]
    yaw = sigma[self._HEAD_YAW_COL, self._HEAD_YAW_COL]
    roll = sigma[self._HEAD_ROLL_COL, self._HEAD_ROLL_COL]

    pitch = np.sqrt(pitch)
    yaw = np.sqrt(yaw)
    roll = np.sqrt(roll)

    return (pitch, roll, yaw)

  def __stddev_face_area(self):
    """ Computes the standard deviation of the face area.
    Returns:
      The standard deviation of the face area. """
    sigma = self.__cov()

    face_area = sigma[self._FACE_AREA_COL, self._FACE_AREA_COL]
    return np.sqrt(face_area)

  def __stddev_face_pos(self):
    """ Computes the standard deviation of the face position.
    Returns:
      The standard deviation of the face position, across y and x. """
    sigma = self.__cov()

    pos_y = sigma[self._FACE_POS_Y_COL, self._FACE_POS_Y_COL]
    pos_x = sigma[self._FACE_POS_X_COL, self._FACE_POS_X_COL]

    pos_y = np.sqrt(pos_y)
    pos_x = np.sqrt(pos_x)

    return (pos_y, pos_x)

  def __corr_pose_accuracy(self):
    """ Computes the correlation between head pose and accuracy.
    Returns:
      The correlation of the head pose, across pitch, yaw, and roll. """
    corr = self.__corr()

    pitch = corr[self._HEAD_PITCH_COL, self._ERROR_COL]
    yaw = corr[self._HEAD_YAW_COL, self._ERROR_COL]
    roll = corr[self._HEAD_ROLL_COL, self._ERROR_COL]

    return (pitch, yaw, roll)

  def __stddev_accuracy_per_subject(self):
    """ Computes the standard deviation of per-subject mean accuracies.
    Returns:
      The standard deviation of per-subject accuracies. """
    subject_error = self.__per_subject_error()

    mean_errors = []
    for error_data in subject_error:
      mean_errors.append(np.mean(error_data))

    return np.std(mean_errors)

  def __corr_area_accuracy(self):
    """ Computes the correlation between face area and accuracy.
    Returns:
      The correlation between face area and accuracy. """
    corr = self.__corr()
    return corr[self._FACE_AREA_COL, self._ERROR_COL]

  def __corr_pos_accuracy(self):
    """ Computes the correlation between face position and accuracy.
    Returns:
      The correlation of the face position, across y and x. """
    corr = self.__corr()

    pos_y = corr[self._FACE_POS_Y_COL, self._ERROR_COL]
    pos_x = corr[self._FACE_POS_X_COL, self._ERROR_COL]

    return (pos_y, pos_x)

  def __plot_per_subject(self):
    """ Generates a box plot of the accuracy on a per-subject level. The plot is
    entered into matplotlib but not shown. """
    subject_error = self.__per_subject_error()

    # Extract session numbers as labels.
    labels = np.unique(self.__data[:, self._SESSION_NUM_COL])
    labels = labels.astype(np.int32)

    plt.boxplot(subject_error, sym="x", labels=labels)
    # Rotate the x labels so they don't get smooshed.
    plt.xticks(rotation=90)

    # Label the axes.
    plt.xlabel("Session Number")
    plt.ylabel("Accuracy (cm)")

  def __write_report(self, report):
    """ Writes a report to the command line. The report is a list, where each
    item is a dictionary. The dictionary should have the following attributes:
      name: The name of the parameter.
      action: 'print' or 'graph', depending on how we want to display it. This
              can also be 'section', in which case just the name is used to
              define a new section in the report.
      value: The value of the parameter.
      unit: Defines a unit for the value.
    Args:
      report: The report to write. """
    print "===== Start of Analysis Report ====="

    for param in report:
      name = param.get("name")
      action = param.get("action")
      value = param.get("value")
      unit = param.get("unit")

      if action == "section":
        # Create a new section.
        print "===== %s =====" % (name.title())

      elif action == "graph":
        # Show the graph.
        plt.title(name.title())
        plt.show()

      elif action == "print":
        # Simply print to command line.
        value_str = str(value)
        separator = " "
        if len(value_str) + len(name) > 80:
          # Split into two lines.
          separator = "\n\t"

        unit_str = ""
        if unit is not None:
          # We have a unit.
          unit_str = " (%s)" % (unit)

        print "%s:%s%s%s" % (name.title(), separator, value_str, unit_str)

    print "===== End of Analysis Report ====="

  def analyze(self):
    """ Performs the full analysis. """
    print "Analyzing..."

    acc_mu = self.__mean_accuracy()
    acc_sigma = self.__stddev_accuracy()
    pitch_mu, yaw_mu, roll_mu = self.__mean_pose()
    pitch_sigma, yaw_sigma, roll_sigma = self.__stddev_pose()
    pitch_corr, yaw_corr, roll_corr = self.__corr_pose_accuracy()

    face_area_mu = self.__mean_face_area()
    face_pos_y_mu, face_pos_x_mu = self.__mean_face_pos()
    face_area_sigma = self.__stddev_face_area()
    face_pos_y_sigma, face_pos_x_sigma = self.__stddev_face_pos()
    face_area_corr = self.__corr_area_accuracy()
    face_pos_y_corr, face_pos_x_corr = self.__corr_pos_accuracy()

    subject_sigma = self.__stddev_accuracy_per_subject()
    self.__plot_per_subject()

    # Write out the report.
    report = [ \
      {"name": "mean error", "action": "print", "value": acc_mu,
       "unit": "cm"},
      {"name": "accuracy standard dev", "action": "print", "value": acc_sigma,
       "unit": "cm"},

      {"name": "head pose", "action": "section"},
      {"name": "head pitch mean", "action": "print", "value": pitch_mu,
       "unit": "rad"},
      {"name": "head roll mean", "action": "print", "value": roll_mu,
       "unit": "rad"},
      {"name": "head yaw mean", "action": "print", "value": yaw_mu,
       "unit": "rad"},
      {"name": "head pitch standard dev", "action": "print",
       "value": pitch_sigma, "unit": "rad"},
      {"name": "head roll standard dev", "action": "print",
       "value": roll_sigma, "unit": "rad"},
      {"name": "head yaw standard dev", "action": "print",
       "value": yaw_sigma, "unit": "rad"},
      {"name": "correlation of head pitch and error", "action": "print",
       "value": pitch_corr},
      {"name": "correlation of head roll and error", "action": "print",
       "value": roll_corr},
      {"name": "correlation of head yaw and error", "action": "print",
       "value": yaw_corr},

      {"name": "face position", "action": "section"},
      {"name": "mean face area", "action": "print", "value": face_area_mu,
       "unit": "gu^2"},
      {"name": "mean face x pos", "action": "print", "value": face_pos_x_mu,
       "unit": "gu"},
      {"name": "mean face y pos", "action": "print", "value": face_pos_y_mu,
       "unit": "gu"},
      {"name": "face area standard dev", "action": "print",
       "value": face_area_sigma, "unit": "gu^2"},
      {"name": "face x pos standard dev", "action": "print",
       "value": face_pos_x_sigma, "unit": "gu"},
      {"name": "face y pos standard dev", "action": "print",
       "value": face_pos_y_sigma, "unit": "gu"},
      {"name": "correlation of face area and error", "action": "print",
       "value": face_area_corr},
      {"name": "correlation of face x pos and error", "action": "print",
       "value": face_pos_x_corr},
      {"name": "correlation of face y pos and error", "action": "print",
       "value": face_pos_y_corr},

      {"name": "per-subject analysis", "action": "section"},
      {"name": "subject mean accuracy standard dev", "action": "print",
       "value": subject_sigma, "unit": "cm"},
      {"name": "per subject accuracy", "action": "graph"},
    ]
    self.__write_report(report)
