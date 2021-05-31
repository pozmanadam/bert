import logging
import sys
import pickle
import torch


if __name__ == '__main__':
  path = "workDir/hun.pkl"
  with open(path, 'wb') as file:
      pickle.dump("/workDir/"+sys.argv[1], file)
