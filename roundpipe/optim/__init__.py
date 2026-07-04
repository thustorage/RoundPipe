"""RoundPipe's implementation of CPU optimizers.
The behavior is consistent to that of PyTorch optimizers,
and optimized for fp32 stepping on CPU.
"""

from .adadelta import Adadelta
from .adafactor import Adafactor
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .nadam import NAdam
from .radam import RAdam
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD

__all__ = [
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
]
