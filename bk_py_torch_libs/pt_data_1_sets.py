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


class DataSetInds(data.Dataset):
    """ A pytorch dataset for indices only (to later grab actual data from dataframes) """
    def __init__(self, inds: tp.Sequence) -> None:
        super().__init__()
        self.inds: torch.LongTensor = to_pytorch_int(inds)

    def __getitem__(self, index):
        return self.inds[index]

    def __len__(self):
        return len(self.inds)
# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE


class DataSetColumnar(data.Dataset):
    """A pytorch dataset for indices only (to later grab data from dataframes)"""
    def __init__(self, cats: np.ndarray = None, conts: np.ndarray = None, y: np.ndarray = None) -> None:
        super().__init__()
        self.cats = cats
        self.conts = conts
        self.y = y
        self.cats, self.conts, self.y  # __i IPYTHON
        self.__len__()  # Asserts all equal

    def __getitem__(self, index):
        # return self.x1[index], self.x2[index], self.x3[index], self.y[index]
        d = dict()
        if self.cats is not None:
            d["cats"] = self.cats[index]
        if self.conts is not None:
            d["conts"] = self.conts[index]
        if self.y is not None:
            d["y"] = self.y[index]
        d
        return d


    def __len__(self):
        return DataSetColumnar.get_shared_lengths(self.cats, self.conts, self.y)
        # lengths = [len(arr) for arr in [self.cats, self.conts, self.y] if arr is not None];  lengths
        # for length in lengths:
        #     assert length == lengths[0]
        # return lengths[0]

    @staticmethod
    def get_shared_lengths(*args: tp.Optional[tp.Sized]):
        """Checks that all non-None *args have equal length, then returns that length"""
        lengths = [len(arg) for arg in args if arg is not None]
        for length in lengths:
            assert length == lengths[0]
        lengths
        return lengths[0]

    @staticmethod
    def fix_types(d: tp.Dict[str, Any]):
        """The data loader usually doesn't get the types right.  This should fix them!"""
        assert d.keys() < set("cats conts y".split())
        if "cats" in d:
            d["cats"] = V(d["cats"]).long()
        if "conts" in d: d["conts"] = V(d["conts"]).float()
        if "y" in d: d["y"] = V(d["y"]).long()
        return d

    @staticmethod
    def TEST_ColumnarDataset():
        """TEST"""
        cats = np.arange(30).reshape((5, 6))
        # conts = cats/10
        y = np.arange(5)
        cats_data_set = DataSetColumnar(cats=cats, y=y)
        cats_data_set[0]
        cats_data_set[1]
        cats_data_set[2]
        assert len(cats_data_set) == 5

        assert DataSetColumnar.get_shared_lengths([2, 3, 4], None, [6, 7, 8]) == 3
