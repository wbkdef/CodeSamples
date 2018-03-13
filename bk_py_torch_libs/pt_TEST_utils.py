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




def get_lin_reg_data(num_each_class: int)  ->  Tuple[torch.FloatTensor, torch.FloatTensor]:
    x = T([[1.0, 0]]*num_each_class + [[0, 1.0]]*num_each_class)
    y = T([0]*num_each_class        + [1.0]*num_each_class)
    x, y
    return x, y
def TEST_get_lin_reg_data() -> None:
    """Simple tests of function "get_lin_reg_data" to help with development in debug mode, as well as for finding bugs"""
    x, y = get_lin_reg_data(2)
    x, y
    assert np.allclose(to_np(x), np.array([[ 1.,  0.],
                                           [ 1.,  0.],
                                           [ 0.,  1.],
                                           [ 0.,  1.]]))
    # __A:  to_np(x)
    print(f"PC:KEY  TEST_get_lin_reg_data done")
    exit(1)
# if __name__ == '__main__':
#     TEST_get_lin_reg_data()

class PassThruDataset(data.Dataset):
    def __init__(self, *xy) -> None:
        super().__init__()
        xy
        self.xys = xy

    def __getitem__(self, index):
        return [xy[index] for xy in self.xys]
        pass

    def __len__(self):
        return len(self.xys[0])
    @staticmethod
    def TEST_PassThruDataset()  ->  None:
        """Simple tests of class "PassThruDataset" to help with development in debug mode, as well as for finding bugs"""
        tensor = T(np.arange(0, 6).reshape((3, 2)))
        ds = PassThruDataset(tensor)
        np.allclose(to_np(ds[1]), [2, 3])
        assert len(ds) == 3

        print(f"PC:KEY  TEST_PassThruDataset done")
        exit(1)
# if __name__ == '__main__':
#     PassThruDataset.TEST_PassThruDataset()

class LinearRegressionZeroed(nn.Module):
    def __init__(self, dims_in: int, dims_out: int):
        super().__init__()
        self.lin = nn.Linear(dims_in, dims_out)
        self.lin.weight.data.zero_() # __c The easy way to set to zero!
        self.lin.bias.data.zero_()
    def forward(self, x: Variable) -> Variable:
        return self.lin(x)
    @staticmethod
    def TEST_LinearRegressionZeroed()  ->  None:
        """Simple tests of class "LinearRegressionZeroed" to help with development in debug mode, as well as for finding bugs"""
        x, y = get_lin_reg_data(3)
        dl = data.DataLoader(PassThruDataset(x, y))
        lrz = to_gpu(LinearRegressionZeroed(2, 1))
        res = lrz(V(x))

        assert np.allclose(*to_np(res.data, res.data * 0))
        lrz.lin
        print(f"PC:KEYyDLt:   TEST_LinearRegressionZeroed done")
        exit(1)
if __name__ == '__main__':
    LinearRegressionZeroed.TEST_LinearRegressionZeroed()
# lr = LinearRegressionZeroed(5, 1)


