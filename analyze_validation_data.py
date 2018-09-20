#!/usr/bin/python


import argparse

from analysis import analyzer


def main():
  # Parse CLI arguments.
  parser = argparse.ArgumentParser(description="Analyze model validation data.")
  parser.add_argument("data_file", help="Validation data input file.")
  args = parser.parse_args()

  my_analyzer = analyzer.Analyzer(args.data_file)
  my_analyzer.analyze()

if __name__ == "__main__":
  main()
