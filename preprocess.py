import collections
import copy

import tensorflow as tf


class Pipeline(object):
  """ A linear sequence of stages that perform operations on an input. """

  def __init__(self):
    # This keeps track of the pipeline output.
    self.__output = None
    # Keeps track of the stages in this pipeline.
    self.__stages = []
    # Keeps track of any pipelines that this one feeds into.
    self.__sub_pipelines = []

  def __build_stage(self, stage):
    """ Builds a single stage of the pipeline.
    Args:
      stage: The stage to build. """
    # For everything but the last stage, we should have only one output.
    assert len(self.__output) == 1
    # Run the stage on our current output.
    outputs = stage.build(self.__output[0])
    if type(outputs) == tf.Tensor:
      # It might have returned a singleton, which we convert to a list.
      outputs = [outputs]

    # Convert output images to datapoints.
    data_points = []
    for output in outputs:
      data_point = copy.copy(self.__output[0])
      data_point.image = output
      data_points.append(data_point)

    self.__output = data_points

  def add(self, stage):
    """ Adds a new stage to the pipeline.
    Args:
      stage: The stage to add.
    Returns:
      If the stage has a single output, the current pipeline is returned.
      Otherwise, the pipeline splits, and multiple new pipelines are
      automatically created and returned. The exact behavior should be specified
      by the pipeline stage. """
    # Add the stage.
    self.__stages.append(stage)
    # Figure out how many outputs we have from this stage.
    num_outputs = stage.get_num_outputs()

    if num_outputs == 1:
      # We can keep using the same pipeline.
      return self

    else:
      # The pipeline forks.
      pipelines = []
      for _ in range(0, num_outputs):
        # Create a new pipeline originating at each output.
        pipeline = Pipeline()
        pipelines.append(pipeline)
        self.__sub_pipelines.append(pipeline)

      return pipelines

  def build(self, data):
    """ Builds the pipeline on a set of input data.
    Args:
      data: The data point to serve as input for the pipeline. """
    # Initially, the output equals the input, in case we have no data.
    self.__output = [data]

    # Build every stage.
    for stage in self.__stages:
      self.__build_stage(stage)

    # Build the sub-pipelines.
    if len(self.__sub_pipelines) > 0:
      for pipeline, output in zip(self.__sub_pipelines, self.__output):
        pipeline.build(output)

  def get_outputs(self):
    """ Gets the ultimate output for this pipeline and any ones downstream. This
    should only be called after build().
    Returns:
      A list of data_points corresponding to the "leaf" outputs from left to
      right. """
    if len(self.__sub_pipelines) == 0:
      # This is the easy case. We just have ourselves to worry about.
      return self.__output

    # In this case, we have to collect the outputs from every sub-pipeline.
    outputs = []
    for pipeline in self.__sub_pipelines:
      outputs.extend(pipeline.get_outputs())
    return outputs

  def get_num_outputs(self):
    """ Gets the total number of outputs from this pipeline and any
    sub-pipelines. This is safe to call at any time.
    Returns:
      The total number of outputs. """
    if len(self.__sub_pipelines) == 0:
      # No sub-pipelines, so we just have our own output.
      return 1

    # Add up the number of outputs from each sub-pipeline.
    num_outputs = 0
    for pipeline in self.__sub_pipelines:
      num_outputs += pipeline.get_num_outputs()
    return num_outputs

class PipelineStage(object):
  """ Defines a stage in the preprocessing pipeline. These can be added
  arbitrarily to data loaders in order to perform preprocessing. """

  def build(self, data_point):
    """ Builds the pipeline stage on a DataPoint object.
    Args:
      data_point: The data_point object to run the stage on.
    Returns:
      The result of the pipeline stage. """
    raise NotImplementedError("build() must be implemented by subclass.")

  def get_num_outputs(self):
    """
    Returns:
      The number of outputs from this pipeline stage. """
    raise NotImplementedError( \
        "get_num_outputs() must be implemented by subclass.")


class RandomCropStage(PipelineStage):
  """ A pipeline stage that extracts a random crop of the image. It has a single
  image output. """

  def __init__(self, crop_size):
    """
    Args:
      crop_size: The size to crop the image at, as (h, w). """
    self.__crop_h, self.__crop_w = crop_size

  def build(self, data_point):
    image = data_point.image

    # Extract the crop.
    num_channels = image.get_shape()[2]
    crop_size = [self.__crop_h, self.__crop_w, num_channels]
    crop = tf.random_crop(image, crop_size)

    return crop

  def get_num_outputs(self):
    return 1

class EyeExtractionStage(PipelineStage):
  """ Extracts eye images from the face crop of the image. It outputs three
  images, in order: The left eye crop, the right eye crop, and the face crop.
  """

  def __convert_box(self, box):
    """ Converts a bounding box from the x, y, w, h format to the y1, x1, y2, x2
    format.
    Args:
      box: The bounding box to convert.
    Returns:
      The converted box. """
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    # Compute the other corners.
    y2 = y + h
    x2 = x + w

    # Create the new tensor.
    return tf.stack([y, x, y2, x2], axis=0)

  def build(self, data_point):
    image = data_point.image
    leye_box = data_point.leye_box
    reye_box = data_point.reye_box

    # Convert the bounding boxes to a form that TensorFlow understands.
    leye_box = self.__convert_box(leye_box)
    reye_box = self.__convert_box(reye_box)
    boxes = tf.stack([leye_box, reye_box], axis=0)

    # Duplicate the input image so that we can crop it twice.
    image_dup = tf.stack([image] * 2, axis=0)

    # Extract the crops using the bounding boxes.
    indices = tf.constant([0, 1])
    # The crops should be resized to the same size as the image.
    crop_size = image.shape[0:2]
    crops = tf.image.crop_and_resize(image_dup, boxes, indices, crop_size)

    leye_crop = crops[0]
    reye_crop = crops[1]

    return (leye_crop, reye_crop, image)

  def get_num_outputs(self):
    return 3

class FaceMaskStage(PipelineStage):
  """ Creates face mask images. It outputs 2 images, in order: The face mask
  image, and the original face crop. """

  def build(self, data_point):
    image = data_point.image
    grid_box = data_point.grid_box

    # The box is in frame fractions initially, so we have to convert it.
    box_sq = grid_box * 25
    box_sq = tf.cast(box_sq, tf.int32)
    # The GazeCapture data is one-indexed. Convert to zero-indexed.
    box_sq -= tf.constant([1, 1, 0, 0])

    # Create the inner section.
    mask_w = box_sq[2]
    mask_h = box_sq[3]
    inner_shape = tf.stack((mask_h, mask_w), axis=0)
    inner = tf.ones(inner_shape, dtype=tf.float32)

    # Compute how much we have to pad by.
    pad_l = box_sq[0]
    pad_r = 25 - (pad_l + mask_w)
    pad_t = box_sq[1]
    pad_b = 25 - (pad_t + mask_h)

    # Pad the inner section to create the mask.
    pad_x = tf.stack((pad_l, pad_r), axis=0)
    pad_y = tf.stack((pad_t, pad_b), axis=0)
    paddings = tf.stack((pad_x, pad_y), axis=0)
    mask = tf.pad(inner, paddings)

    return (mask, image)

  def get_num_outputs(self):
    return 2
