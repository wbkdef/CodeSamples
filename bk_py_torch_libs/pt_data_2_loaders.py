import os
import os.path as osp
import re
import sys
import enum
import typing as tp
import itertools as it
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable, Sized, NamedTuple, Optional
from functools import partial

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F  # noinspection PyPep8Naming
import torch.utils.data as data
import torch.optim

# import bk_ds_libs.test_py_torch_helpers as pth
from bk_py_torch_libs import pt_core
from bk_py_torch_libs.pt_core import T, V, VV, to_np, to_gpu

from bk_ds_libs import bk_utils_ds as utd
from bk_ds_libs import bk_data_sets


import bk_general_libs.bk_typing as tp_bk
from bk_general_libs.bk_typing import SelfSequence, SelfList, SelfTuple, SelfIterable, SelfList_Recursive, SelfSequence_Recursive, SelfIterable_Recursive, NonNegInt, NonNegFloat, Probability, NNI, NNF, PBT, TV
from bk_general_libs import bk_itertools
from bk_general_libs import bk_decorators
from bk_general_libs import bk_strings


class DataLoadersColumnar:
    """Class for managing train/valid data loaders for columnar data"""
    # pylint: disable=too-many-arguments
    def __init__(self, frac_valid=.2, cats: np.ndarray = None, conts: np.ndarray = None, y: np.ndarray = None, batch_size=1, valid_batch_size: int = None,
                 shuffle=True) -> None:
        """

        :param cats:
        :param conts:
        :param y:
        :param batch_size:
        :param valid_batch_size: If "None", batch_size is used for the validation set data loader as well
        :param shuffle:
        """
        super().__init__()
        self._shuffle = shuffle
        # self._y = y
        # self._conts = conts
        # self._cats = cats
        all_data: tp.Dict = dict(cats=cats, conts=conts, y=y)
        all_input_data = {name: arr for name, arr in all_data.items() if arr is not None}

        # lengths = DataSetColumnar.get_shared_lengths(self._cats, self._conts, self._y)
        lengths = DataSetColumnar.get_shared_lengths(*all_input_data.values())
        self._inds_train, self._inds_valid = train_test_split(np.arange(lengths), test_size=frac_valid)

        self._train_input_data = {name: arr[self._inds_train] for name, arr in all_input_data.items()}
        self._valid_input_data = {name: arr[self._inds_valid] for name, arr in all_input_data.items()}

        train_dataset = DataSetColumnar(**self._train_input_data)
        valid_dataset = DataSetColumnar(**self._valid_input_data)

        self.train_loader = data.DataLoader(train_dataset, batch_size, shuffle)
        valid_batch_size = valid_batch_size or batch_size
        self.valid_loader = data.DataLoader(valid_dataset, valid_batch_size, shuffle)

    # __c Create tests for this!

    @staticmethod
    def TEST_DataLoadersColumnar():
        """TEST"""
        cats = np.arange(60).reshape((10, 6))
        # conts = cats/10
        y = np.arange(10)
        loaders = DataLoadersColumnar(cats=cats, y=y, batch_size=3)
        for batch in loaders.train_loader:
            print(f"\n train batch is [[{batch}]]")
        for batch in loaders.valid_loader:
            print(f"\n valid batch is [[{batch}]]")

        loaders
# if __name__ == '__main__':
#     DataLoadersColumnar.TEST_DataLoadersColumnar()