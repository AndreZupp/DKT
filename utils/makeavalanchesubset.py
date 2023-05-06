import copy
import warnings

from torch.utils.data.dataloader import default_collate

from avalanche.benchmarks.utils.dataset_definitions import IDataset
from  avalanche.benchmarks.utils import DataAttribute

from typing import List, Any, Sequence, Union, TypeVar, Callable

from .flat_data import FlatData
from .transform_groups import TransformGroups, EmptyTransformGroups
from torch.utils.data import Dataset as TorchDataset


