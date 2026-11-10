from .naive import NaiveStrategy
from .ewc import EWCStrategy
from .lwf import LwFStrategy
from .er import ERStrategy
from .joint import JointStrategy
from .fixed import FixedLambdaStrategy, OracleStrategy
from .knn import KNNStrategy

STRATEGIES = {
    "naive": NaiveStrategy,
    "ewc": EWCStrategy,
    "lwf": LwFStrategy,
    "er": ERStrategy,
    "joint": JointStrategy,
    "fixed": FixedLambdaStrategy,
    "oracle": OracleStrategy,
    "knn": KNNStrategy,
}
