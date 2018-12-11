import logging

from ..common import custom_data_loader
from ..pipeline import keras_utils, preprocess


logger = logging.getLogger(__name__)


class PipelineBuilder(object):
  """ Responsible for building and configuring input pipelines. """

  def __init__(self, raw_shape, image_size, batch_size, eye_size=None,
               tpu_flatten=False):
    """
    Args:
      raw_shape: The original shape we expect for images loaded from the disk.
      image_size: The size to use for output images from the pipeline, as a
                  tuple of (h, w).
      batch_size: The size of the batches to load.
      eye_size: The size to use for output eye images, if this is different from
                the one for face images.
      tpu_flatten: Whether to use the flat outputs for the TPU. """
    self.__image_size = image_size
    self.__raw_shape = raw_shape
    self.__batch_size = batch_size
    self.__tpu_flatten = tpu_flatten

    # Calculate sizes for cropping and resizing.
    self.__face_resize_to = [int(dim / 0.975) for dim in self.__image_size]
    self.__eye_resize_to = [int(dim / 0.9) for dim in self.__image_size]
    logger.debug("Initial face resize: %s" % (str(self.__face_resize_to)))
    logger.debug("Initial eye resize: %s" % (str(self.__eye_resize_to)))

    self.__eye_size = self.__image_size
    if eye_size is not None:
      self.__eye_size = eye_size

  def __fuse_loaders(self, train_loader, train_pipelines, test_loader,
                     test_pipelines):
    """ Fuses the outputs from the training and testing loaders.
    Args:
      train_loader: The training loader.
      train_pipelines: The pipelines associated with the train loader.
      test_loader: The testing loader.
      test_pipelines: The pipelines associated with the test loader.
    Returns:
      The fused outputs, in the same order as the pipeline inputs, with the labels
      at the end. """
    train_data = train_loader.get_data()
    train_labels = train_loader.get_labels()
    test_data = test_loader.get_data()
    test_labels = test_loader.get_labels()

    # Extract the corresponding outputs for the pipelines.
    train_outputs = []
    for pipeline in train_pipelines:
      train_outputs.append(train_data[pipeline])
    # Add the labels too.
    train_outputs.append(train_labels)

    test_outputs = []
    for pipeline in test_pipelines:
      test_outputs.append(test_data[pipeline])
    test_outputs.append(test_labels)

    # Fuse the outputs.
    return keras_utils.fuse_loaders(train_outputs, test_outputs)

  def __add_train_stages(self, loader, has_pose, has_session_num):
    """ Convenience function to configure train loader.
    Args:
      loader: The DataLoader to configure.
      has_pose: Whether our dataset contains head pose data.
      has_session_num: Whether we have a session number input. """
    pipeline = loader.get_pipeline()

    # Extract eye crops.
    extract_stage = preprocess.EyeExtractionStage(eye_size=self.__eye_resize_to)
    leye, reye, face = pipeline.add(extract_stage)

    # Extract face mask.
    mask_stage = preprocess.FaceMaskStage()
    mask, face = face.add(mask_stage)

    # Resizing.
    face_resize_stage = preprocess.ResizeStage(self.__face_resize_to)
    face.add(face_resize_stage)

    # Random cropping.
    crop_stage = preprocess.RandomCropStage(self.__eye_size)
    face_crop_stage = preprocess.RandomCropStage(self.__image_size)
    leye.add(crop_stage)
    reye.add(crop_stage)
    face.add(face_crop_stage)

    # Random adjustments.
    brightness_stage = preprocess.RandomBrightnessStage(50)
    contrast_stage = preprocess.RandomContrastStage(0.9, 1.4)
    hue_stage = preprocess.RandomHueStage(0.1)
    saturation_stage = preprocess.RandomSaturationStage(0.9, 1.1)
    grayscale_stage = preprocess.GrayscaleStage()

    #leye.add(brightness_stage)
    #leye.add(contrast_stage)
    #leye.add(grayscale_stage)

    #reye.add(brightness_stage)
    #reye.add(contrast_stage)
    #reye.add(grayscale_stage)

    #face.add(brightness_stage)
    #face.add(contrast_stage)
    #face.add(hue_stage)
    #face.add(saturation_stage)

    # Normalization.
    norm_stage = preprocess.NormalizationStage()

    #leye.add(norm_stage)
    #reye.add(norm_stage)
    #face.add(norm_stage)

    # Session number stage.
    if has_session_num:
      session_num_stage = preprocess.SessionNumStage()
      session_num, face = face.add(session_num_stage)

      session_num.associate_with_input("session_num_input")

    if has_pose:
      # Pose extraction.
      pose_stage = preprocess.HeadPoseStage()
      pose, face = face.add(pose_stage)

      pose.associate_with_input("pose_input")

    # Name the pipelines.
    leye.associate_with_input("left_eye_input")
    reye.associate_with_input("right_eye_input")
    face.associate_with_input("face_input")
    mask.associate_with_input("grid_input")

    # Build the loader graph.
    loader.build()

  def __add_test_stages(self, loader, has_pose, has_session_num):
    """ Convenience function to configure test and validation loaders.
    Args:
      loader: The DataLoader to configure.
      has_pose: Whether our dataset contains head pose data.
      has_session_num: Whether our model has a session number input. """
    pipeline = loader.get_pipeline()

    # Extract eye crops.
    extract_stage = preprocess.EyeExtractionStage()
    leye, reye, face = pipeline.add(extract_stage)

    # Extract face mask.
    mask_stage = preprocess.FaceMaskStage()
    mask, face = face.add(mask_stage)

    # Take the central crops.
    crop_stage = preprocess.CenterCropStage(0.975)
    face_crop_stage = preprocess.CenterCropStage(0.9)
    leye.add(crop_stage)
    reye.add(crop_stage)
    face.add(face_crop_stage)

    # Convert to grayscale.
    grayscale_stage = preprocess.GrayscaleStage()
    leye.add(grayscale_stage)
    reye.add(grayscale_stage)

    # Normalization and final sizing.
    norm_stage = preprocess.NormalizationStage()
    face_resize_stage = preprocess.ResizeStage(self.__image_size)
    eye_resize_stage = preprocess.ResizeStage(self.__eye_size)

    leye.add(norm_stage)
    reye.add(norm_stage)
    face.add(norm_stage)

    leye.add(eye_resize_stage)
    reye.add(eye_resize_stage)
    face.add(face_resize_stage)

    # Session number stage.
    if has_session_num:
      session_num_stage = preprocess.SessionNumStage()
      session_num, face = face.add(session_num_stage)

      session_num.associate_with_input("session_num_input")

    if has_pose:
      # Pose extraction.
      pose_stage = preprocess.HeadPoseStage()
      pose, face = face.add(pose_stage)

      pose.associate_with_input("pose_input")

    # Name the pipelines.
    leye.associate_with_input("left_eye_input")
    reye.associate_with_input("right_eye_input")
    face.associate_with_input("face_input")
    mask.associate_with_input("grid_input")

    # Build the loader graph.
    loader.build()

  def build_pipeline(self, train_data, test_data, has_pose=False,
                     has_session_num=False):
    """ Builds the preprocessing pipeline.
    Args:
      train_data: The training data TFRecords file.
      test_data: The testing data TFRecords file.
      has_pose: Whether or not the model has a head pose input.
      has_session_num: Whether or not the model has a session number input.
    Returns:
      The fused output nodes from the loaders, in order: leye, reye, face, grid,
      dots. """
    train_loader_class = custom_data_loader.TrainDataLoader
    test_loader_class = custom_data_loader.TestDataLoader
    if has_pose:
      # Use loaders with pose attribute support.
      train_loader_class = custom_data_loader.TrainDataLoaderWithPose
      test_loader_class = custom_data_loader.TestDataLoaderWithPose

    train_loader = train_loader_class(train_data, self.__batch_size,
                                      self.__raw_shape,
                                      tpu_flatten=self.__tpu_flatten)
    test_loader = test_loader_class(test_data, self.__batch_size,
                                    self.__raw_shape,
                                    tpu_flatten=self.__tpu_flatten)

    train_pipelines = self.__add_train_stages(train_loader, has_pose,
                                              has_session_num)
    test_pipelines = self.__add_test_stages(test_loader, has_pose,
                                            has_session_num)

    return train_loader.get_data()
    #return self.__fuse_loaders(train_loader, train_pipelines,
    #                           test_loader, test_pipelines)

  def build_valid_pipeline(self, valid_data, has_pose=False,
                           has_session_num=False):
    """ Builds the preprocessing pipeline for the validation split.
    Args:
      valid_data: The validation data TFRecords file.
      has_pose: Whether or not a head pose attribute is included in the data.
      has_session_num: Whether or not the model has a session number input.
    Returns:
      The leye, reye, face, grid, and dots nodes for the validation loader. """
    valid_loader_class = custom_data_loader.ValidDataLoader
    if has_pose:
      # Use loader with pose attribute support.
      valid_loader_class = custom_data_loader.ValidDataLoaderWithPose

    valid_loader = valid_loader_class(valid_data, self.__batch_size,
                                      self.__raw_shape,
                                      tpu_flatten=self.__tpu_flatten)

    # Use the same pipeline for validation as we do for testing.
    valid_pipelines = self.__add_test_stages(valid_loader, has_pose,
                                             has_session_num)

    # Extract the associated output nodes.
    data = valid_loader.get_data()
    nodes = []
    for pipeline in valid_pipelines:
      nodes.append(data[pipeline])
    nodes.append(valid_loader.get_labels())

    return nodes
