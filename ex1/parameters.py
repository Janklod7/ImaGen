import torch.nn as nn
"""
data parameters
"""

# path to image directory
DATA_PATH = "C:/dataset/dataset/"

# dataset sizes:
TRAIN_SIZE = 1000

VALIDATION_SIZE = 1000


"""
Hyper parameters
"""

# leaky relu allow rate
LEAKY_RATE = 0.1


# number of epoches
EPOCHS = 30

# increasing batch size and learning rates together
INCREASE = 2
# batch size
BATCH_SIZE = 4*INCREASE

# learning rate
LEARNING_RATE = 0.001*INCREASE

criterion = nn.MSELoss()

LAYER1 = 64
LAYER2 = 16
LAYER3 = 4
LAYER4 = 1  # DO NOT CHANGE THIS!!!

"""
running parameters
"""

# autoencoder from different file
# replace to your chosen autoencoder
from shachar_simple import ConvAutoencoder as model    # change this line

MODEL = model()    # !!! dont change this line !!!

# include debug prints from debug_print
DEBUG = False

# change to true to run a small batch and print sizes
DEBUG_SIZES = False

# use gpu via cuda
USE_GPU = True


"""
accesories
"""


def debug_print(*args, mode=DEBUG):
    """
    debugs only if DEBUG = True
    :param mode: the debug mode for that print
    :param args: arguments for printing
    """
    if mode:
        print(*args)


def printsize(i, x):
    debug_print(i, x[0].size(), mode=DEBUG_SIZES)
    return x


if DEBUG_SIZES:
    TRAIN_SIZE = BATCH_SIZE
    VALIDATION_SIZE = BATCH_SIZE

