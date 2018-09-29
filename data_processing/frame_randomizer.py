import random


class FrameRandomizer(object):
  """ Class that stores and randomizes frame data. """

  def __init__(self):
    # Create dictionary of sessions.
    self.__sessions = []
    self.__total_examples = 0

    # This is a list of indices representing all sessions in the dataset in
    # random order.
    self.__random_sessions = None

  def __build_random_sessions(self):
    """ Builds the random sessions list after all sessions have been added. """
    self.__random_sessions = []

    for i, session in enumerate(self.__sessions):
      self.__random_sessions.extend([i] * session.num_valid())
      # Shuffle the data in the session.
      session.shuffle()

    # Shuffle all of them.
    random.shuffle(self.__random_sessions)

  def add_session(self, session):
    """ Add data for one session.
    Args:
      session: The session to add. """
    self.__total_examples += session.num_valid()
    self.__sessions.append(session)

  def get_random_example(self):
    """ Draws a random example from the session pool. It raises a ValueError if
    there is no more data left.
    Returns:
      The next random example, including the features and extracted face crop,
      in the following order: crop, bytes features, float features, int
      features. """
    if self.__random_sessions is None:
      # Build the session pick list.
      self.__build_random_sessions()

    if (len(self.__sessions) == 0 or len(self.__random_sessions) == 0):
      # No more data.
      raise ValueError("Session pool has no more data.")

    # First, pick a random session.
    session_key = self.__random_sessions.pop()
    session = self.__sessions[session_key]

    # Now, pick a random example from within that session.
    crop, bytes_f, float_f, int_f = session.get_random()

    return (crop, bytes_f, float_f, int_f)

  def get_num_examples(self):
    """
    Returns:
      The total number of examples. """
    return self.__total_examples
