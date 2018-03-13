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

from bk_ds_libs import bk_utils_ds as utd
from bk_ds_libs import bk_data_sets

import bk_general_libs.bk_typing as tp_bk
from bk_general_libs.bk_typing import SelfSequence, SelfList, SelfTuple, SelfIterable, SelfList_Recursive, SelfSequence_Recursive, SelfIterable_Recursive, NonNegInt, NonNegFloat, Probability, NNI, NNF, PBT, TV
from bk_general_libs import bk_itertools
from bk_general_libs import bk_decorators
from bk_general_libs import bk_strings


# noinspection PyProtectedMember
TensorOrVariable = tp.Union[torch._TensorBase, Variable]
Type = tp.TypeVar('Type')
ItselfOrListOf = tp.Union[Type, tp.List[Type]]
TensorOrVariableOrListOf = ItselfOrListOf[TensorOrVariable]
ArrayLike = tp.Iterable  # Usually a list, a numpy array, etc., that want to convert to a torch.Tensor/Variable

tests_to_run = []


# __t Jeremy's numpy/pytorch converters from:  C:\Users\wbruc\Desktop\git_repos\fast.ai2\fastai\core.py
# noinspection PyArgumentList
# pylint: disable=invalid-name
def T(arr: ArrayLike)  ->  tp.Union[torch.cuda.LongTensor, torch.cuda.FloatTensor]:
    """Converts input to arr torch Long/Float tensor, usually on the GPU"""
    if torch.is_tensor(arr): res = arr
    else:
        arr_np: np.ndarray = np.array(np.ascontiguousarray(arr))
        if arr_np.dtype in (np.bool, np.int8, np.int16, np.int32, np.int64):
            res = torch.LongTensor(arr_np.astype(np.int64))
        elif arr_np.dtype in (np.float32, np.float64):
            res = torch.FloatTensor(arr_np.astype(np.float32))
        else:
            raise NotImplementedError(f"numpy type of not recognized for arr_np: {arr_np}")
    if isinstance(res, (torch.IntTensor, torch.cuda.IntTensor)):
        res = res.long()
    elif isinstance(res, (torch.DoubleTensor, torch.cuda.DoubleTensor)):
        res = res.float()
    assert isinstance(res, (torch.LongTensor, torch.cuda.LongTensor, torch.FloatTensor, torch.cuda.FloatTensor))

    # noinspection PyTypeChecker
    return to_gpu(res, async=True)
# pylint: disable=invalid-name
def TEST_T():
    """Tests for function "T" """
    double_tensor = torch.from_numpy(np.arange(5) / 5)
    # double_tensor.dtype
    res2 = T(double_tensor)
    assert isinstance(res2, torch.cuda.FloatTensor)
    to_np(res2) == np.arange(5)/5
    res2

    float_tensor = torch.from_numpy((np.arange(5) / 5).astype('float32'))
    res2 = T(float_tensor)
    assert isinstance(res2, torch.cuda.FloatTensor)
    to_np(res2) == np.arange(5)/5
    res2

    double_range = np.arange(5) / 5
    double_range.dtype
    res2 = T(double_range)
    assert isinstance(res2, torch.cuda.FloatTensor)
    to_np(res2) == np.arange(5)/5
    res2

    float_range = np.arange(5) / 5
    float_range = float_range.astype('float32')
    double_range.dtype
    res2 = T(float_range)
    assert isinstance(res2, torch.cuda.FloatTensor)
    to_np(res2) == np.arange(5)/5
    res2

    res = T(range(5))
    assert isinstance(res, torch.cuda.LongTensor)
    to_np(res) == np.arange(5)
    res
# pylint: disable=unnecessary-lambda
tests_to_run.append(lambda: TEST_T())


def create_variable(x: ArrayLike, volatile=False, requires_grad=False)  ->  Variable:
    """Converts x to a Tensor, then to a Variable, usually on the GPU"""
    if not isinstance(x, Variable):
        x = Variable(T(x), volatile=volatile, requires_grad=requires_grad)
    # return to_gpu(x, async=True)
    return x

def V_(x: ArrayLike, requires_grad=False)  ->  Variable:
    """Converts x to a Tensor, then to a Variable, usually on the GPU"""
    return create_variable(x, False, requires_grad=requires_grad)

# @tp.overload
# def V(x: List[ArrayLike])  ->  List[Variable]: pass
# @tp.overload
# def V(x: ArrayLike)  ->  Variable: pass
def V(x: ItselfOrListOf[ArrayLike], requires_grad=False)  ->  Variable:
    """Applies V_ to x or, if x is a list, to each element of x.  REQUIRES_GRAD FALSE BY DEFAULT!

    V_ Converts x to a Tensor, then to a Variable, usually on the GPU"""
    return [V_(o, requires_grad) for o in x] if isinstance(x, list) else V_(x, requires_grad)

def VV_(x: ArrayLike)  ->  Variable:
    """Converts x to a Tensor, then to a Variable, with volatile=True!!!, usually on the GPU"""
    return create_variable(x, volatile=True)

# @tp.overload
# def VV(x: List[ArrayLike])  ->  List[Variable]: pass
# @tp.overload
# def VV(x: ArrayLike)  ->  Variable: pass
def VV(x: ItselfOrListOf[ArrayLike])  ->  ItselfOrListOf[Variable]:
    """Converts x or List[x] to a Tensor, then to a Variable, with volatile=True!!!, usually on the GPU"""
    return [VV_(o) for o in x] if isinstance(x, list) else VV_(x)

# @tp.overload  # __c This is giving an error message I don't understand - also, PyCharm less helpful when overloaded, cause just gives first defn back!
# def to_np(values: TensorOrVariable)  ->  np.ndarray: pass
# @tp.overload
# def to_np(values: List[TensorOrVariable])  ->  List[np.ndarray]: pass
def to_np(*values: SelfList[Union[TensorOrVariable, str]])  ->  SelfList[np.ndarray]:
    """Converts a Tensor/Variable/List-thereof to numpy array(s) on the CPU"""
    if len(values) == 0: raise ValueError("At least 1 argument required!")
    if len(values) == 1: values = values[0]  # get rid of the list it's packed in!
    if isinstance(values, (list, tuple)): return [to_np(o) for o in values]
    if isinstance(values, Variable): values = values.data
    if isinstance(values, str): return _str_to_np(values)
    return values.cpu().numpy()

from bk_general_libs import bk_utils
def _str_to_np(s: str)  ->  np.ndarray:
    lines = s.splitlines()
    rows = [re.split("[\s,]+", line.strip()) for line in lines if line.strip() != '']
    # __c Could also use "max(rows, key=len)", but might as well compare the numbers, rather than the values.
    assert min(map(len, rows)) == max(map(len, rows))
    arr = np.array([[bk_utils.to_num(num) for num in row]  for row in rows])
    return arr


def TEST__str_to_np() -> None:
    """Simple tests of function "_str_to_np" to help with development in debug mode, as well as for finding bugs"""
    res = _str_to_np("""
    1 2
    3   4
    5 9
    """)
    correct = np.array([[1, 2], [3, 4], [5, 9]])
    assert res.dtype == correct.dtype
    assert np.allclose(res, correct)
    print(f"PC:KEY  TEST__str_to_np done")
    exit(1)
if __name__ == '__main__':
    TEST__str_to_np()

USE_GPU = True
def to_gpu(x: TensorOrVariableOrListOf, *args, **kwargs)  ->  Any:
    """Usually puts the Variable/Tensor (or list thereof) "x" on the gpu (using arguments *args, **kwargs)

    If the gpu is not available or USE_GPU == False, then it won't be put on the GPU
    """
    if isinstance(x, (list, tuple)): return [to_gpu(o, *args, **kwargs) for o in x]
    if torch.cuda.is_available() and USE_GPU:
        return x.cuda(*args, **kwargs)
    return x


def noop(*args, **kwargs): return


def trainable_params_(m: nn.Module)  ->  List[nn.Parameter]:
    """ Extract, from an 'nn.Module', a list of just the trainable parameters """
    return [p for p in m.parameters() if p.requires_grad]

def chain_trainable_params(p: Union[nn.Module, List, Tuple])  ->  List[nn.Parameter]:
    """Extracts out all trainable parameters from an nn.Module or a list/tuple thereof

    todo Refactor so accepts any iterable (just checks if isinstance(nn.Module) first)?
    todo Replace the chain with summing over lists
    """
    if isinstance(p, (list,tuple)):
        return list(it.chain(*[trainable_params_(o) for o in p]))
    return trainable_params_(p)


# def set_trainable_attr(m: nn.Module,b: bool):
#     """Saves you doing a for loop through the parameters to set requires_grad"""
#     m.trainable=b  # __c This doesn't seem to be used in PyTorch, or in fast.ai!
#     for p in m.parameters(): p.requires_grad=b

# Don't expect to need this - This is for use with his learner class, which maybe doesn't have the .modules functionality.
# def apply_leaf(m, f):
#     c = children(m)
#     if isinstance(m, nn.Module): f(m)
#     if len(c)>0:
#         for l in c: apply_leaf(l,f)

# def set_trainable(l, b):
#     apply_leaf(l, lambda m: set_trainable_attr(m,b))

# Used by learner
# def SGD_Momentum(momentum):
#     return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

# def one_hot(a,c): return np.eye(c)[a]

class TEST_ALL:

    def __init__(self) -> None:
        super().__init__()
        self.TEST_split_by_idxs()

    def TEST_split_by_idxs(self):
        res = list(split_by_idxs(list(range(6)), (2, 5)))
        assert res == [[0, 1], [2, 3, 4], [5]]



if __name__ == '__main__':
    TEST_ALL()
    for test in tests_to_run:
        test()
