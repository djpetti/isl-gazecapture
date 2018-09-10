#!/usr/bin/python


import argparse

from itracker.training import experiment

import logging_config


# How many iterations to validate for.
valid_iters = 124


def parse_args():
  """ Builds a parser for CLI arguments.
  Returns:
    The parser it built. """
  parser = argparse.ArgumentParser()

  parser.add_argument("train_dataset",
                      help="The location of the training dataset.")
  parser.add_argument("test_dataset",
                      help="The location of the testing dataset.")

  parser.add_argument("-v", "--valid_dataset",
                      help="The location of the validation dataset.")
  parser.add_argument("-m", "--model",
                      help="Existing model to load. Necessary if validating.")
  parser.add_argument("-o", "--output", default="eye_model.hd5",
                      help="Where to save the trained model.")

  parser.add_argument("--batch_size", type=int, default=32,
                      help="Examples in each batch.")
  parser.add_argument("--testing_interval", type=int, default=4,
                      help="No. of training iterations to run before testing.")
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="The initial learning rate.")
  parser.add_argument("--momentum", type=float, default=0.9,
                      help="The initial momentum.")
  parser.add_argument("--training_steps", type=int, default=40,
                      help="Num. of batches to run for each training iter.")
  parser.add_argument("--testing_steps", type=int, default=24,
                      help="Num. of batches to run for each testing iter.")

  return parser

def validate(args):
  """ Validates an existing model.
  Args:
    args: Parsed CLI arguments. """
  if not args.valid_dataset:
    raise ValueError("--valid_dataset must be specified.")

  # Create the validation pipeline.
  face_size = config.FACE_SHAPE[:2]
  eye_size = config.EYE_SHAPE[:2]
  builder = pipelines.PipelineBuilder(raw_shape, face_size, batch_size,
                                      eye_size=eye_size)

  input_tensors = builder.build_valid_pipeline(args.valid_dataset)
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  input_tensors = build_valid_pipeline(args)
  data_tensors = input_tensors[:4]
  label_tensor = input_tensors[4]

  # Create the model.
  net = config.NET_ARCH(config.FACE_SHAPE, eye_shape=config.EYE_SHAPE,
                        data_tensors=data_tensors)
  model = net.build()
  load_model = args.model
  if not load_model:
    # User did not tell us which model to validate.
    raise ValueError("--model must be specified.")
  logging.info("Loading pretrained model '%s'." % (load_model))
  model.load_weights(load_model)

  # Compile the model. The learning settings don't really matter, since we're
  # not training.
  opt = optimizers.SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss=distance_metric, metrics=[distance_metric],
                target_tensors=[label_tensor])

  # Create a coordinator and run queues.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=session)

  testing_acc = []

  # Validate.
  for _ in range(0, valid_iters):
    loss, accuracy = model.evaluate(steps=test_interval)

    logging.info("Loss: %f, Accuracy: %f" % (loss, accuracy))
    testing_acc.append(accuracy)

  print "Total accuracy: %f" % (np.mean(testing_acc))

  coord.request_stop()
  coord.join(threads)

def main():
  logging_config.configure_logging()
  parser = parse_args()

  # Create and start the experiment.
  my_experiment = experiment.Experiment(parser)
  my_experiment.train()


if __name__ == "__main__":
  main()
