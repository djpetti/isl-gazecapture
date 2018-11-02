import cPickle as pickle

import matplotlib.pyplot as plt

import numpy as np

import sklearn.cluster

import analyzer_base


class Analyzer(analyzer_base.AnalyzerBase):
  """ Handles analysis for the autoencoder validation data. """

  # Constants defining the indices of the variable columns in the data matrix.
  _GAZE_ERROR_COL = 0
  _DECODE_ERROR_COL = 1
  _HEAD_PITCH_COL = 2
  _HEAD_YAW_COL = 3
  _HEAD_ROLL_COL = 4
  _SESSION_NUM_COL = 5
  _ENCODING_START_COL = 6

  def __init__(self, data_file, num_clusters, cluster_file):
    """
    Args:
      data_file: The file to load the validation data from.
      num_clusters: How many clusters to break encoded data into.
      cluster_file: File to save cluster data to. """
    super(Analyzer, self).__init__(data_file)

    self.__num_clusters = num_clusters
    self.__cluster_file = cluster_file
    # Clustering instance to use for K-means clustering.
    self.__clustering = sklearn.cluster.KMeans(self.__num_clusters, n_jobs=-1,
                                               max_iter=2000)

  def __fit_clusters(self):
    """ Computes the clustering for the encodings. """
    print "Performing K-means clustering..."

    # Extract the encodings from the larger matrix.
    encodings = self._data[:, self._ENCODING_START_COL:]
    # Fit with the clustering algorithm.
    self.__clustering.fit(encodings)

    print "Done!"

  def __divide_clusters(self):
    """ Divides points by clusters. """
    self.__cluster_points = {}

    # Predict clusters for all points.
    encodings = self._data[:, self._ENCODING_START_COL:]
    clusters = self.__clustering.predict(encodings)

    for i in range(0, clusters.shape[0]):
      # Mark the point as part of the cluster.
      cluster = int(clusters[i])
      if cluster not in self.__cluster_points:
        self.__cluster_points[cluster] = []
      self.__cluster_points[cluster].append(self._data[i])

    # Convert each cluster list to a numpy array.
    for cluster, points in self.__cluster_points.iteritems():
      self.__cluster_points[cluster] = np.stack(points, axis=0)

  def __per_subject_clustering(self):
    """ Determines which cluster each subject falls into.
    Returns:
      A dictionary, mapping session numbers to cluster indices. """
    clusters = {}
    for cluster, points in self.__cluster_points.iteritems():
      for i in range(0, points.shape[0]):
        point = points[i]
        # Mark every session number as belonging to this cluster.
        session_num = int(point[self._SESSION_NUM_COL])

        if session_num not in clusters:
          clusters[session_num] = set()
        clusters[session_num].add(cluster)

    return clusters

  def __per_cluster_gaze_error(self):
    """ Determines the average gaze error for each cluster.
    Returns:
      A dictionary, mapping clusters to error values. """
    error = {}

    for cluster, points in self.__cluster_points.iteritems():
      mean_error = np.mean(points[:, self._GAZE_ERROR_COL])
      error[cluster] = mean_error

    return error

  def __per_cluster_dec_error(self):
    """ Determines the average decode error for each cluster.
    Returns:
      A dictionary, mapping clusters to error values. """
    error = {}

    for cluster, points in self.__cluster_points.iteritems():
      mean_error = np.mean(points[:, self._DECODE_ERROR_COL])
      error[cluster] = mean_error

    return error

  def __cluster_size(self):
    """ Determines the number of points in each cluster.
    Returns:
      A dictionary, mapping clusters to sizes. """
    cluster_sizes = {}

    for cluster, points in self.__cluster_points.iteritems():
      size = points.shape[0]
      cluster_sizes[cluster] = size

    return cluster_sizes

  def __per_cluster_head_pose(self):
    """ Determines the average head pose for each cluster.
    Returns:
      A dictionary, mapping clusters to head pose in the order pitch, yaw, roll.
    """
    pitch = {}
    yaw = {}
    roll = {}

    for cluster, points in self.__cluster_points.iteritems():
      mean_pitch = np.mean(points[:, self._HEAD_PITCH_COL])
      mean_yaw = np.mean(points[:, self._HEAD_YAW_COL])
      mean_roll = np.mean(points[:, self._HEAD_ROLL_COL])

      pitch[cluster] = mean_pitch
      yaw[cluster] = mean_yaw
      roll[cluster] = mean_roll

    return (pitch, yaw, roll)

  def __save_clusters(self):
    """ Saves the learned clusters. """
    out_file = open(self.__cluster_file, "wb")
    pickle.dump(self.__clustering.cluster_centers_, out_file)
    out_file.close()

  def analyze(self):
    # Perform the clustering, which is computationally-intensive.
    self.__fit_clusters()
    self.__divide_clusters()
    self.__save_clusters()

    cluster_centers = self.__clustering.cluster_centers_
    center_norms = [np.linalg.norm(x) for x in cluster_centers]

    subject_clusters = self.__per_subject_clustering()
    cluster_gaze_errors = self.__per_cluster_gaze_error()
    cluster_dec_errors = self.__per_cluster_dec_error()
    cluster_sizes = self.__cluster_size()

    pitch, yaw, roll = self.__per_cluster_head_pose()

    # Save the cluster data.
    self.__save_clusters

    # Write out the report.
    report = [ \
      {"name": "cluster sizes", "action": "print", "value": cluster_sizes},
      {"name": "cluster norms", "action": "print", "value": center_norms},
      {"name": "subject clusters", "action": "print",
       "value": subject_clusters},
      {"name": "cluster gaze errors", "action": "print",
       "value": cluster_gaze_errors},
      {"name": "cluster dec errors", "action": "print",
       "value": cluster_dec_errors},
      {"name": "head pitch", "action": "print", "value": pitch},
      {"name": "head roll", "action": "print", "value": roll},
      {"name": "head yaw", "action": "print", "value": yaw},
    ]
    self._write_report(report)
