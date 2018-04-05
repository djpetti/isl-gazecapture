from yaml import load, dump
try:
	from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
	from yaml import Loader, Dumper


class PhoneConfig(object):
  """ Represents physical phone configuration. """

  def __init__(self, config_file):
    """
    Args:
      config_file: The path to the configuration file for the phone. """
    # Load the data.
    data_file = file(config_file)
    self.__data = load(data_file, Loader=Loader)
    data_file.close()

  def get_screen_cm(self):
    """ Returns: The size of the screen in cm. """
    long_cm = self.__data["ScreenData"]["ScreenLongCm"]
    short_cm = self.__data["ScreenData"]["ScreenShortCm"]

    return (long_cm, short_cm)

  def get_resolution(self):
    """ Returns: The screen resolution. """
    res_long = self.__data["ScreenData"]["ResolutionLong"]
    res_short = self.__data["ScreenData"]["ResolutionShort"]

    return (res_long, res_short)

  def get_camera_offset(self):
    """ Returns: The camera offset in cm. """
    offset_long = self.__data["CameraData"]["CameraLongOffset"]
    offset_short = self.__data["CameraData"]["CameraShortOffset"]

    return (offset_long, offset_short)

  def get_camera_fov(self):
    """ Returns: The camera FOV in cm. """
    fov_long = self.__data["CameraData"]["CameraFOVLong"]
    fov_short = self.__data["CameraData"]["CameraFOVShort"]

    return (fov_long, fov_short)
