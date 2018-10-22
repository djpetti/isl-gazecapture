import cPickle as pickle

import matplotlib.pyplot as plt


class AnalyzerBase(object):
  """ Base class for analyzers. """

  def __init__(self, data_file):
    """
    Args:
      data_file: The file to load the validation data from. """
    self._load_data(data_file)

  def _load_data(self, data_file):
    """ Loads the data from a file. """
    print "Loading validation data..."
    self._data = pickle.load(file(data_file, "rb"))

  def _write_report(self, report):
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
    raise NotImplementedError("analyze() must be implemented by subclass.")
