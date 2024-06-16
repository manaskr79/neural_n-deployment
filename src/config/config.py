import sys
import pathlib
import os

# Add the src directory to the Python path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import src

NUM_LAYERS = 3
layer_sizes = [2, 2, 1]

f = ["", "tanh", "sigmoid"]

LOSS_FUNCTION = "Mean Squared Error"
MINI_BATCH_SIZE = 2
tol = 10**-8 

epsilon = 10**-7

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")