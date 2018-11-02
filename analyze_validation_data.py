#!/usr/bin/python


import argparse

from analysis import analyzer, autoencoder_analyzer


def main():
  # Parse CLI arguments.
  parser = argparse.ArgumentParser(description="Analyze model validation data.")
  parser.add_argument("data_file", help="Validation data input file.")
  parser.add_argument("-a", "--autoencoder", action="store_true",
                      help="If set, use special autoencoder validation.")
  parser.add_argument("-k", "--num_clusters", type=int, default=2,
                      help="Number of clusters to use for autoencoder data.")
  parser.add_argument("-o", "--output", default="analyzer_output.pkl",
                      help="Output file to use.")
  args = parser.parse_args()

  my_analyzer = None
  if args.autoencoder:
    # Use autoencoder analyzer.
    my_analyzer = autoencoder_analyzer.Analyzer(args.data_file,
                                                args.num_clusters,
                                                args.output)
  else:
    # Use normal analyzer.
    my_analyzer = analyzer.Analyzer(args.data_file)

  my_analyzer.analyze()

if __name__ == "__main__":
  main()
