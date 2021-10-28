"""
data parameters
"""

# path to image directory
DATA_PATH = "C:/dataset/dataset/"

# dataset sizes:
VALIDATION_SIZE = 1000
TRAIN_SIZE = 100

"""
Hyper parameters
"""

# leaky relu allow rate
LEAKY_RATE = 0.1

# number of epoches
EPOCHS = 100

# batch size
BATCH_SIZE = 4

# learning rate
LEARNING_RATE = 0.001

"""
running parameters
"""

# autoencoder from different file
# replace to your chosen autoencoder
from AE_w_linear import ConvAutoencoder as model    # change this line

MODEL = model()    # !!! dont change this line !!!

# include debug prints from debug_print
DEBUG = False

# change to true to run a small batch and print sizes
DEBUG_SIZES = True

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

