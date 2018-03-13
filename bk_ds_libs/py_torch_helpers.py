"""
import bk_ds_libs.test_py_torch_helpers as pth

# done PyLint-C 2018-02-10
"""
# __t Standard Header

import os
import os.path as osp
import re
import sys
import typing as tp
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable
import enum
import shelve

import bk_ds_libs.bk_data_processing
import bk_general_libs.bk_caching
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F  # noinspection PyPep8Naming
import torch.utils.data as data
import torch.optim

from bk_ds_libs import bk_utils_ds as utd
from bk_ds_libs import bk_data_sets
# from bk_envs.env_general import *

import bk_general_libs.bk_typing as tp_bk
from bk_general_libs.bk_typing import SelfSequence, SelfList, SelfIterable, SelfList_Recursive, SelfSequence_Recursive, SelfIterable_Recursive
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


# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE
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

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

def create_variable(x: ArrayLike, volatile=False, requires_grad=False)  ->  Variable:
    """Converts x to a Tensor, then to a Variable, usually on the GPU"""
    if not isinstance(x, Variable):
        x = Variable(T(x), volatile=volatile, requires_grad=requires_grad)
    # return to_gpu(x, async=True)
    return x

def V_(x: ArrayLike, requires_grad=False)  ->  Variable:
    """Converts x to a Tensor, then to a Variable, usually on the GPU"""
    return create_variable(x, False, requires_grad=requires_grad)
# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

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
# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

# @tp.overload  # __c This is giving an error message I don't understand - also, PyCharm less helpful when overloaded, cause just gives first defn back!
# def to_np(values: TensorOrVariable)  ->  np.ndarray: pass
# @tp.overload
# def to_np(values: List[TensorOrVariable])  ->  List[np.ndarray]: pass
def to_np(values: SelfList[TensorOrVariable])  ->  SelfList[np.ndarray]:
    """Converts a Tensor/Variable/List-thereof to numpy array(s) on the CPU"""
    if isinstance(values, (list, tuple)): return [to_np(o) for o in values]
    if isinstance(values, Variable): values = values.data
    return values.cpu().numpy()


USE_GPU = True
def to_gpu(x: TensorOrVariableOrListOf, *args, **kwargs)  ->  Any:
    """Usually puts the Variable/Tensor (or list thereof) "x" on the gpu (using arguments *args, **kwargs)

    If the gpu is not available or USE_GPU == False, then it won't be put on the GPU
    """
    if isinstance(x, (list, tuple)): return [to_gpu(o, *args, **kwargs) for o in x]
    if torch.cuda.is_available() and USE_GPU:
        return x.cuda(*args, **kwargs)
    return x

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

# __t My original converters
# noinspection PyProtectedMember
def _to_pytorch_tensor(arr: ArrayLike)  ->  torch._TensorBase:
    if not isinstance(arr, torch._TensorBase):
        # pylint: disable=broad-except
        # noinspection PyBroadException
        try:
            # Try to convert from pandas to numpy
            arr = arr.values  # type: ignore
        except Exception:
            pass
        
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        arr = torch.from_numpy(arr)
    assert isinstance(arr, torch._TensorBase)
    return arr


# noinspection PyProtectedMember
def to_pytorch_int(arr: ArrayLike) -> torch.LongTensor:
    """Converts 'arr' to a torch.LongTensor"""
    tensor = _to_pytorch_tensor(arr)
    tensor = tensor.long()
    return tensor
# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE


# noinspection PyProtectedMember
def to_pytorch_float(arr: ArrayLike) -> torch.FloatTensor:
    """Converts 'arr' to a torch.FloatTensor"""
    tensor = _to_pytorch_tensor(arr)
    tensor = tensor.float()
    return tensor

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

# __t More Stuff
# noinspection PyStatementEffect
class DataSource:
    """Groups some categorical and continuous data and info so it can be embedded into a neural network"""
    # pylint: disable=missing-docstring, too-many-arguments
    name: str

    dx_cats: pd.DataFrame
    dx_conts: pd.DataFrame
    inds: utd.Ind

    embedding_size: int
    embed: bool
    on_gpu: bool

    def __init__(self, inds: utd.Indexes, dx_cats: pd.DataFrame = None, dx_conts: pd.DataFrame = None, embed=True, embedding_size=None, on_gpu=True, name=None) -> None:
        super().__init__()
        # assert dx_cats.values.dtype == 'int64'
        # assert dx_conts.values.dtype == 'float32'

        self.name = name
        self.dx_cats = dx_cats
        self.dx_conts = dx_conts
        self.inds = inds

        self.embedding_size = embedding_size
        self.embed = embed

        # __c Maybe this should already be on the GPU?  Will this be slowing it down a bunch?
        # NO!  I think they generally put it on the GPU inside their loops.  Particularly when using pictures.
        self.on_gpu = on_gpu

    # __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

    # These get items, given indices
    def _get_loc(self, inds: utd.Ind, inside_inds: utd.Ind)  ->  tp.Tuple[torch.LongTensor, torch.FloatTensor]:
        """Returns items with given index"""
        assert set(inds) < set(inside_inds)
        df_cats: pd.DataFrame = self.dx_cats.loc[inds]
        df_conts: pd.DataFrame = self.dx_conts.loc[inds]
        t_cats: torch.LongTensor = to_pytorch_int(df_cats)
        t_conts: torch.FloatTensor = to_pytorch_float(df_conts)
        if self.on_gpu:
            t_cats.cuda()
            t_conts.cuda()
        return t_cats, t_conts
    def get_train(self, inds)  ->  tp.Tuple[torch.LongTensor, torch.FloatTensor]:  return self._get_loc(inds, self.inds.train)
    def get_test(self, inds)   ->  tp.Tuple[torch.LongTensor, torch.FloatTensor]:  return self._get_loc(inds, self.inds.test)
    def get_valid(self, inds)  ->  tp.Tuple[torch.LongTensor, torch.FloatTensor]:  return self._get_loc(inds, self.inds.val)

    def get_combined_df(self)  ->  pd.DataFrame:
        """The purpose of this is to be able to nicely display the CATS and CONTS together"""
        if self.dx_cats is None:
            return self.dx_conts
        if self.dx_conts is None:
            return self.dx_cats
        joined = pd.merge(self.dx_cats, self.dx_conts, 'outer', left_index=True, right_index=True, suffixes=('CAT', 'CTS')); joined
        return joined

    def __repr__(self, rows=10) -> str:
        s = f"DataSource {self.name}\n"
        s += f"inds: {self.inds}\n"
        if self.embed:
            s += f"embedding_size: {self.embedding_size}"
        else:
            s += 'NOT EMBEDDED'
        s += f'\non_gpu: {self.on_gpu}\n'
        s += f'{self.get_combined_df()[:rows]}'
        s += '\n...'
        s += f'{self.get_combined_df()[-rows:]}'
        return s
        # return super().__str__()

    # __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

    @staticmethod
    def from_cat(inds: utd.Indexes,
                 s: pd.Series,
                 embedding_size: int,
                 own_group_threshold=20,
                 name: str = None):
        """Generates a data source from a single categorical variable (provided as a pd.Series)

        :param inds:
        :param s:
        :param embedding_size:
        :param own_group_threshold: A threshold below which items are mapped to "0", rather than forming their own category
        :param name: A name for the data_source, to help with debugging later.
        :return:
        """
        processed: pd.DataFrame = pd.DataFrame(utd.CatVar(s, inds, own_group_threshold=own_group_threshold).s_out); processed
        return DataSource(inds, dx_cats=processed, embedding_size=embedding_size, name=name)

    @staticmethod
    def from_cont(inds: utd.Indexes,
                  s: pd.Series,
                  embedding_size: int,
                  own_group_threshold=20,
                  bins: tp.Sequence = None,
                  name: str = None):
        """Generates a data source from a single continuous variable (provided as a pd.Series)

        :param inds:
        :param s:
        :param embedding_size:
        :param own_group_threshold: A threshold below which items are mapped to "0", rather than forming their own category
        :param bins: Optional interpolated binning centers of the continuous variable
        :param name: A name for the data_source, to help with debugging later.
        :return:
        """
        processed: pd.DataFrame = utd.ContVar(s, inds, own_group_threshold=own_group_threshold, bins=bins).df_out; processed
        return DataSource(inds, dx_conts=processed, embedding_size=embedding_size, name=name)

    # __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

    @staticmethod
    def TEST_DataSource():
        """Test method for DataSource"""
        df = bk_data_sets.DataSetTitanic.get_combined()
        ind_test = df[df.Survived.isnull()].index
        dt, dv = train_test_split(df[~df.Survived.isnull()], test_size=.25)
        ind_train = dt.index
        ind_val = dv.index
        inds = utd.Indexes(ind_train, ind_val, ind_test)

        cat_sex: pd.DataFrame = pd.DataFrame(utd.CatVar(df.Sex, inds, 20).s_out); cat_sex
        cont_age: pd.DataFrame = utd.ContVar(df.Age, inds, 20, [0, 5, 12, 25, 100]).df_out; cont_age[:2]
        data_source = DataSource(inds, cat_sex, cont_age, embedding_size=4, name="titanic-age&sex")

        data_source.get_combined_df()
        print(data_source)
# DataSource.TEST_DataSource()


class DataSetInds(data.Dataset):
    """A pytorch dataset for indices only (to later grab data from dataframes)"""
    def __init__(self, inds: tp.Sequence) -> None:
        super().__init__()
        self.inds: torch.LongTensor = to_pytorch_int(inds)

    def __getitem__(self, index):
        return self.inds[index]

    def __len__(self):
        return len(self.inds)
# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE


@bk_decorators.deprecated
def IndsDataset(inds: tp.Sequence)  ->  DataSetInds:
    """Making the old name for DataSetInds still work in old code"""
    return DataSetInds(inds)


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

    # __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

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
# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE


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

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

class DataSourceEmbedder(nn.Module):
    """Embeds a single instance of DataSource"""
    def __init__(self, ds: DataSource) -> None:
        """Smart embedding, combining cats and continuous, then putting through a nonlinearity (leaky relu?)
        Embed each, individually, into size "embedding_size" so that dimensionality is not reduced more than it might ultimately be,
        Then Concat and Linear & Nonlinearity into dimension embedding_size

        Possibly add more layers (in residual style) later
        Possibly treat a single categorical special
        """
        super().__init__()
        self.ds: DataSource = ds

        df_cats = self.ds.dx_cats
        embedding_size = self.ds.embedding_size

        df_cats
        # col = 'Sex'
        self.embeddings: List[nn.Embedding] = []
        tot_embeddings_size: int = 0
        if df_cats is not None:
            for col in df_cats:
                s = df_cats[col]
                int(s.max())
                emb = nn.Embedding(int(s.max()) + 1, embedding_size)
                tot_embeddings_size += embedding_size
                self.embeddings.append(emb)
                self._modules[f"{self.ds.name}_{col}"] = emb
        if self.ds.dx_conts is not None:
            tot_embeddings_size += len(self.ds.dx_conts.columns)

        self.combiner = nn.Linear(tot_embeddings_size, embedding_size)
        self.combiner

    @property
    def output_size(self) -> int:
        """The size of torch.FloatTensor it will output"""
        return self.ds.embedding_size

    def forward(self, inds):
        to_concatenate = []
        if self.ds.dx_cats is not None:
            df_cat = self.ds.dx_cats.loc[inds]
            df_cat  # __i IPYTHON

            for col, nn_emb in zip(df_cat, self.embeddings):
                cats = Variable(to_pytorch_int(df_cat[col])).cuda()
                cats
                to_concatenate.append(nn_emb(cats))

        if self.ds.dx_conts is not None:
            cont: torch.FloatTensor = to_pytorch_float(self.ds.dx_conts.loc[inds])
            to_concatenate.append(Variable(cont).cuda())

        concatenated = torch.cat(to_concatenate, 1)

        x = self.combiner(concatenated)
        x = F.leaky_relu(x, .5)
        x  # __i IPYTHON
        return x

    @staticmethod
    def TEST_DataSourceEmbedder():
        """Tests this class"""
        pass
        # inds, dss, survived = do_titanic_data_processing_and_get_data_sources()
        #
        # # __c To help program in debug mode
        # emb = DataSourceEmbedder(dss[0])
        # emb.cuda()
        # res_cat = emb(range(5))
        #
        # emb = DataSourceEmbedder(dss[5])
        # emb.cuda()
        # res_cont = emb(inds.get_rand_inds_train(7))
        #
        # res_cat
        # res_cont, dss[5].name


# DataSourceEmbedder.TEST_DataSourceEmbedder()

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

class DataSourcesCombiner(nn.Module):
    """Embeds and concatenates a bunch of instances of DataSource.

    Individual data sources may be randomly dropped out"""
    def __init__(self, dss: tp.Iterable[DataSource])  ->  None:
        super().__init__()
        self.embeddings: tp.List[DataSourceEmbedder] = []
        self.tot_embeddings_size: int = 0
        for i, ds in enumerate(dss):
            # # __c CONTINUE HERE
            self.embeddings.append(DataSourceEmbedder(ds))
            self.tot_embeddings_size += ds.embedding_size
            self._modules[f"{ds.name}_{i}"] = self.embeddings[-1]
        self.tot_embeddings_size  # __i IPYTHON
        len(self.embeddings)

    @property
    def output_size(self) -> int:
        """The size of torch.FloatTensor .forward()[0] returns"""
        return self.tot_embeddings_size

    @property
    def dropouts_records_size(self) -> int:
        """The size of torch.FloatTensor .forward()[1] returns"""
        return len(self.embeddings)

    def forward(self, inds: tp.Sequence, sources_dropout_rate):
        to_concatenate = []
        sources_dropouts = np.random.rand(len(inds), len(self.embeddings)) < sources_dropout_rate;  sources_dropouts  # if True, then dropout
        sources_dropouts = to_pytorch_float(sources_dropouts.astype('float32'))  # if 1, then dropout
        sources_dropouts = Variable(sources_dropouts).cuda()
        sources_dropouts  # __i IPYTHON
        # Implement dropping out or amping up of sources
        # Should record in advance what will be dropped out, which is returned
        # Could also record the dropout rate
        # i = 0  # __i IPYTHON
        for i, nn_emb in enumerate(self.embeddings):
            out = nn_emb(inds)
            dropout = sources_dropouts[:, i].unsqueeze(1);  dropout
            out_dropped = out * (1 - dropout);  out_dropped
            out  # __i IPYTHON
            to_concatenate.append(out_dropped)

        concatenated = torch.cat(to_concatenate, 1)
        concatenated, sources_dropouts
        assert concatenated.size()[1] == self.tot_embeddings_size

        return concatenated, sources_dropouts

    @staticmethod
    def TEST_DataSourcesCombiner():
        """Tests this class"""
        pass
        # inds, dss, survived = do_titanic_data_processing_and_get_data_sources()
        #
        # com = DataSourcesCombiner(dss)
        # com.cuda()
        # # com(range(4))
        # res = com(inds.get_rand_inds_train(7), .2)
        # res
# DataSourcesCombiner.TEST_DataSourcesCombiner()

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

class SAMPLE_Net(nn.Module):
    """A sample neural network - that was used for the titanic"""
    def __init__(self, dss: tp.Iterable[DataSource], num_classes)  ->  None:
        super(SAMPLE_Net, self).__init__()
        # self.num_classes = num_classes
        self.data_sources_combiner = DataSourcesCombiner(dss)

        in_size = self.data_sources_combiner.output_size + self.data_sources_combiner.dropouts_records_size
        out_size = self.data_sources_combiner.tot_embeddings_size

        # self.concat_with_dropout_rec = torch.cat
        self.lin1 = nn.Linear(in_size, out_size)
        self.bn1 = nn.BatchNorm1d(out_size)
        self.relu1 = nn.LeakyReLU(.5)

        self.do_f = nn.Dropout(.5)
        # self.concat_with_dropout_rec
        self.lin_f = nn.Linear(in_size, num_classes)
        self.softmax = nn.LogSoftmax()

        # self.do1 = nn.Dropout(.1)

    def forward(self, inds: torch.LongTensor, sources_dropout_rate)  ->  torch.FloatTensor:
        x, dropout_rec = self.data_sources_combiner.forward(inds, sources_dropout_rate)

        x = torch.cat([x, dropout_rec], 1)
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.do_f(x)
        x = torch.cat([x, dropout_rec], 1)
        x = self.lin_f(x)
        preds = self.softmax(x)
        preds
        return preds

    @staticmethod
    def TEST_Net():
        """Tests this class"""
        pass
        # inds, dss, survived = do_titanic_data_processing_and_get_data_sources()
        #
        # net = SAMPLE_Net(dss, 3)
        # net.cuda()
        # # net(range(4))
        # res = net(inds.get_rand_inds_train(7), .2)
        # print(f"\n res is [[{res}]]")
# SAMPLE_Net.TEST_Net()

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

class CharRnnFOR_TESTS(nn.Module):
    """A simple char RNN model, to use for testing other parts of this module"""
    def __init__(self, vocab_size, n_fac, n_hidden, num_x_to_use):
        super().__init__()
        self.num_x_to_use = num_x_to_use
        self.n_hidden = n_hidden
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)

    def forward(self, x: Variable)  ->  Variable:
        x = x[:, -self.num_x_to_use:];  x
        x
        cs: tp.List[torch.cuda.LongTensor] = [x[:, i] for i in range(x.data.shape[1])];  cs

        batch_size = cs[0].size(0)

        h = V(torch.zeros(1, batch_size, self.n_hidden))
        stacked = torch.stack(cs, 0)  # __c First dimension should be the sequence of the rnn, 2nd is the batch!
        stacked
        inp = self.e(stacked)
        inp
        # __c inp should be a tensor of shape: (seq_len, batch, input_size)
        # __c "h" defaults to 0
        outp, h = self.rnn(inp, h)
        outp, h
        # return F.log_softmax(self.l_out(outp[-1]), dim=-1)
        return F.log_softmax(self.l_out(outp[-1]))

    # @staticmethod
    # def TEST_CharRnn():
    #     m = CharRnnFOR_TESTS(vocab_size=14, n_fac=5, n_hidden=10, num_x_to_use=7)
    #     m = to_gpu(m)
    #     x = V(torch.stack([torch.arange(0, 12), torch.arange(2, 14)], 0)).long();  x
    #     x
    #     print(f"\nm(x) is [[{m(x)}]]")
    #     exit(1)
    #     x
# if __name__ == '__main__':
#     CharRnn.TEST_CharRnn()


# __c A starter for when creating something new!
# def do_SAMPLE_TRAINING_LOOP():
#     inds, dss, survived = do_titanic_data_processing_and_get_data_sources()
#     survived
#     inds.get_rand_inds_train(7)
#     dss
#
#     train_dataset = IndsDataset(inds.train)
#     valid_dataset = IndsDataset(inds.val)
#     train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     valid_loader = data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)
#
#     net: SAMPLE_Net = SAMPLE_Net(dss, 3)
#     net.cuda()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
#
#     source_dropout_max_rate = .5
#
#     # Train the Model
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for i, b_inds in enumerate(train_loader):
#             # todo2 Make to_pytorch_float take care of Variable and .cuda(), by default.  Also, take care of boolean numpy arrays at the same time!  I guess can just change the numpy type!
#             b_inds_np = b_inds.numpy()
#             by: Variable = Variable(to_pytorch_int(survived[b_inds_np])).cuda();  by
#
#             source_dropout_rate = random.uniform(0, source_dropout_max_rate)
#
#             # Forward + Backward + Optimize
#             optimizer.zero_grad()  # zero the gradient buffer
#             outputs = net(b_inds_np, source_dropout_rate)
#             loss = criterion(outputs, by)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.data
#             if (i + 1) % 10 == 0:
#                 print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                       % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, epoch_loss[0] / (i + 1)))
#                 # % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
#
#         epoch_loss = epoch_loss.cpu()[0] / len(train_loader)
#         print(f"PC:KEYlLeY:  epoch_loss: {epoch_loss}", end=",   ")
#
#         for i, b in enumerate(valid_loader):
#             bx_cat: torch.IntTensor = b[0]; bx_cat[:10]
#             bx_cont: torch.DoubleTensor = b[1:-1]; bx_cont[:10]
#             by = b[-1]
#
#             bx_cat = Variable(bx_cat).cuda()
#             bx_cont = [Variable(bx).cuda() for bx in bx_cont]
#             by = Variable(by).cuda()
#
#             outputs = net(bx_cat, bx_cont)
#             loss = criterion(outputs, by)
#             # noinspection PyArgumentList
#             _, predicted = torch.max(outputs.data, 1)
#             preds_right: np.ndarray = predicted.cpu().numpy() == by.data.cpu().numpy()
#             correct = sum(preds_right)
#             total = len(preds_right)
#             print(f'Valid accuracy: {(100*correct/total):.0%},    loss: {loss.data[0]:.2f}')
#
#     torch.save(net.state_dict(), 'titanic_1hidden_model.pkl')

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE


# __t Memoizing model weights with multiple versions
@enum.unique
class WeightOption(enum.Enum):
    """Parameter options for class ModelWeightsMemoizer's methods"""
    latest = enum.auto()
    latest_incremented = enum.auto()

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

class ModelWeightsMemoizer:
    """Class for saving versions of a model's weights, and easily reloading the latest (or any specified version)."""
    net: nn.Module
    _pickle_manager: bk_general_libs.bk_caching.PickleManager
    _shelf_filename: str

    def __init__(self, net: nn.Module, pickle_manager: bk_general_libs.bk_caching.PickleManager) -> None:
        super().__init__()
        self.net = net

        self._pickle_manager = pickle_manager
        self._shelf_filename = pickle_manager.get_filename(f'shelf')

    def load_weights_into_net(self, weights_pickle_num: Union[int, WeightOption] = WeightOption.latest)  ->  int:
        """Loads the model's weights - from the latest save, by default"""
        weights_pickle_num = self._get_weights_pickle_num(weights_pickle_num)
        weights_file_name: str = self.get_weights_file_name(weights_pickle_num)
        try:
            self.net.load_state_dict(torch.load(weights_file_name))
            print(f"PC:KEYqoBC:  Loaded weights from {weights_file_name}")
            # torch.load(weights_file_name)
        except FileNotFoundError as e:
            print(f"PC:KEYsarV:  Weights file doesn't exist.  Starting from random weights. e: {e}, {type(e)}")
        return weights_pickle_num

    def get_weights_file_name(self, weights_pickle_num: Union[int, WeightOption])  ->  str:
        """Generates the appropriate filename using
                weights_pickle_num
                net.__class__.__name__
                self._pickle_manager
        """
        self._get_weights_pickle_num(weights_pickle_num)
        weights_file_name = self._pickle_manager.get_filename(f'{self.net.__class__.__name__}{weights_pickle_num}')
        return weights_file_name

    def _get_weights_pickle_num(self, weights_pickle_num: Union[int, WeightOption])  ->  int:
        """Convenience function - turns weights_pickle_num into the appropriate integer if a WeightOption was given"""
        if weights_pickle_num in [WeightOption.latest, WeightOption.latest_incremented]:
            increment: bool = (weights_pickle_num is WeightOption.latest_incremented)
            weights_pickle_num = self.get_latest_weights_pickle_num(increment=increment)
        assert isinstance(weights_pickle_num, int)
        return weights_pickle_num

    def get_latest_weights_pickle_num(self, *, increment: bool)  ->  int:
        """ Gets the latest pickle num

        :param increment: Whether to increment the latest pickle num before returning
        :return:
        """
        with shelve.open(self._shelf_filename) as db:
            if 'weights_pickle_num' in db:
                weights_pickle_num = db['weights_pickle_num']
            else:
                weights_pickle_num = 4
            if increment:
                weights_pickle_num += 1
                db['weights_pickle_num'] = weights_pickle_num
        return weights_pickle_num

    def save_net_weights(self, weights_pickle_num: Union[int, WeightOption] = WeightOption.latest_incremented)  ->  int:
        """ Saves the model's weights, incrementing the pickle num, by default

        :param weights_pickle_num:
        :return:
        """
        weights_pickle_num = self._get_weights_pickle_num(weights_pickle_num)
        weights_file_name: str = self.get_weights_file_name(weights_pickle_num)
        torch.save(self.net.state_dict(), weights_file_name)
        print(f"PC:KEYqoBC:  Saved weights to {weights_file_name}")
        return weights_pickle_num

    @staticmethod
    def TEST_ModelWeightsMemoizer():
        """TEST"""
        net = CharRnnFOR_TESTS(86, 40, 100, 9)
        PM = bk_general_libs.bk_caching.PickleManager("pickles", "nietszche_rnn_char_model", data_subset_size=None)
        model_weight_memoizer = ModelWeightsMemoizer(net, PM)
        model_weight_memoizer.load_weights_into_net()
        model_weight_memoizer.save_net_weights()
        model_weight_memoizer.load_weights_into_net()
        model_weight_memoizer.save_net_weights()
        net
        print(f"PC:KEYbVNH:  Done testing ModelWeightsMemoizer")
        exit(1)
        pass
# if __name__ == '__main__':
#     ModelWeightsMemoizer.TEST_ModelWeightsMemoizer()

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

# __t Generating text (or sequences of digits) with models that work on recurrent data
class SequentialModelGenerator:
    """Uses a NN on sequential integer data to generate new integers from a starting sequence."""
    def __init__(self, net: Callable[[Variable], Variable], *, min_x_to_use: int) -> None:
        """
        :param net: A Neural Net that
              ->  takes:    Data representing a sequence of min_x_to_use integers - Variable(batch size X min_x_to_use)
              ->  returns:  Data representing the probabilities of the next integer value - Variable(batch size X num_possible_ints)
        :param min_x_to_use: The minimum sequence length of integers needed for the model to predict the next integer
        """
        super().__init__()
        self.net = net
        self.min_x_to_use = min_x_to_use

    def get_next_probs(self, idxs: Sequence[int]) -> np.ndarray:
        """Generate the probabilities of each possible value of the next index after "idxs"

        :param idxs:
        :return: 1D np.ndarray of length the # of possible integers
        """
        log_probs = self.get_next_log_probs(idxs)
        probs = np.exp(log_probs)
        assert abs(np.sum(probs) - 1) < .001
        return probs

    def get_next_log_probs(self, idxs: Sequence[int]) -> np.ndarray:
        """Generate the log of the probabilities of each possible value of the next index after "idxs"

        :param idxs:
        :return: 1D np.ndarray of length the # of possible integers
        """
        assert len(idxs) >= self.min_x_to_use
        idxs = idxs[-self.min_x_to_use:]
        idxs = np.array(idxs)[None]  # Convert to numpy and Add the batch dimension
        idxs
        net_input = V(idxs)
        log_probs: np.ndarray = to_np(self.net(net_input))
        assert len(log_probs) == 1
        assert len(log_probs.shape) == 2
        log_probs = log_probs[0]
        return log_probs

    def generate_next_idx_most_likely(self, idxs: Sequence[int]) -> int:
        """Generate the next index after "idxs", where this index is the most likely as predicted by the net (on idxs)"""
        lp = self.get_next_log_probs(idxs)
        idx = lp.argmax()
        return idx

    def generate_next_idx_probabilistically(self, idxs: Sequence[int]) -> int:
        """Generate the next index after "idxs", where the probability of each index being generated is as predicted by the net (on idxs)

        :param idxs:
        :return:
        """
        # raise NotImplementedError
        lp = self.get_next_probs(idxs)
        lp
        idx = np.random.choice(np.arange(len(lp)), p=lp)
        idx
        # idx = lp.argmax()
        return idx

    def generate_next_n_idxs_most_likely(self, starter_idxs: Iterable[int], *, n: int, include_original_idxs=True) -> List[int]:
        """Generate the next n indices, where each index is that of highest probability, as calculated by the net

        :param starter_idxs:
        :param n:
        :param include_original_idxs: Whether to return the starter_idxs in the list of indices returned
        :return:
        """
        idxs = self.generate_next_n_idxs(starter_idxs, pred_fcn=self.generate_next_idx_most_likely, n=n, include_original_idxs=include_original_idxs)
        return idxs

    def generate_next_n_idxs_probabilistically(self, starter_idxs: Iterable[int], *, n: int, include_original_idxs=True) -> List[int]:
        """Generate the next n indices, where the probability of each index being generated is as predicted by the net (on the previous characters)

        :param starter_idxs:
        :param n:
        :param include_original_idxs: Whether to return the starter_idxs in the list of indices returned
        :return:
        """
        idxs = self.generate_next_n_idxs(starter_idxs, pred_fcn=self.generate_next_idx_probabilistically, n=n, include_original_idxs=include_original_idxs)
        return idxs

    @staticmethod
    def generate_next_n_idxs(starter_idxs: Iterable[int], *, pred_fcn: Callable[[Sequence[int]], int], n: int, include_original_idxs=True) -> List[int]:
        """Helper method for 1) generating n new indices, 2) appending them to a copy of the starter_idxs (if include_original_idxs=True), and 3) returning them

        :param starter_idxs: Starter indices - neural network requires some indices to calculate the probabilities of the next.
        :param pred_fcn: Method for choosing the next index.  Generall "generate_next_idx_most_likely" OR "generate_next_idx_probabilistically"
        :param n:
        :param include_original_idxs: Switch - should starter indices be included in return indices?
        :return: [starter_idxs +] generated indices
        """
        idxs: List[int] = list(starter_idxs)
        start_len = len(idxs)
        # pylint: disable=unused-variable
        for i in range(n):
            pred = pred_fcn(idxs)
            idxs.append(pred)
        if not include_original_idxs:
            idxs = idxs[start_len:]
        return idxs

    # __c Would need to make simple model for testing this (and for next).  Currently using in: n2018_01_18_lesson_6_nietzsche/n4.1_rnn/l6_5_nietszche_test_model.py:34
    # @staticmethod
    # def TEST_SequentialModelGenerator() -> None:
    #     net = m_model.get_model()
    #     model_weight_memoizer = ModelWeightsMemoizer(net, setup.PM)
    #     model_weight_memoizer.load_weights_into_net()
    #
    #     predictor = SequentialModelGenerator(net, min_x_to_use=net.num_x_to_use)
    #
    #     start_idxs = list(range(20))
    #     # probs = predictor.get_next_probs(start_text)
    #     gen_max_likelihood = predictor.generate_next_n_idxs_most_likely(start_idxs, n=100)
    #     gen_probabilistic = predictor.generate_next_n_idxs_probabilistically(start_idxs, n=100)
    #     gen_max_likelihood
    #     gen_probabilistic
    #
    #     print(f"PC:KEYfchv:  TEST_SequentialModelGenerator done")
    #     exit(1)


# if __name__ == '__main__':
#     SequentialModelGenerator.TEST_SequentialModelGenerator()

# __c STOP ADDING TO THIS LIB!  NOW COPYING FUNCTIONALITY TO THE PYTORCH LIBS PACKAGE

class CharacterModelGenerator:
    """Uses a NN and a TextData (to map between characters and integers) to generate characters from a starting string."""
    text_data: bk_ds_libs.bk_data_processing.TextData
    sequential_integer_model: SequentialModelGenerator

    def __init__(self, net: Callable[[Variable], Variable], text_data: bk_ds_libs.bk_data_processing.TextData, *, min_x_to_use: int) -> None:
        """

        :param net:  Generally a nn.Model
        :param text_data:  TextData (to map between characters and integers)
        :param min_x_to_use:  Min chars needed to feed into the model (to predict the next char)
        """
        super().__init__()
        self.text_data = text_data
        self.sequential_integer_model = SequentialModelGenerator(net, min_x_to_use=min_x_to_use)

    def generate_next_n_most_likely(self, starter_chars: str, *, n: int, include_starter_chars=True) -> str:
        """Generate next n characters, where each character is that of highest probability, as calculated by the net
        :param starter_chars:  Starter string - neural network requires some characters to calculate the probabilities of the next.
        :param n:  Num new chars to generate
        :param include_starter_chars:  Switch - should starter chars be included in return string?
        :return:  String of [starter_chars +] generated chars
        """
        idxs: List[int] = self.text_data.convert_str_to_idxs(starter_chars)
        idxs_out: List[int] = self.sequential_integer_model.generate_next_n_idxs_most_likely(idxs, n=n, include_original_idxs=include_starter_chars)
        res = self.text_data.convert_idxs_to_str(idxs_out)
        return res

    def generate_next_n_probabilistically(self, starter_chars: str, *, n: int, include_starter_chars=True) -> str:
        """Generate next n characters, where the probability of each character is calculated by the net
        :param starter_chars:  Starter string - neural network requires some characters to calculate the probabilities of the next.
        :param n:  Num new chars to generate
        :param include_starter_chars:  Switch - should starter chars be included in return string?
        :return:  String of [starter_chars +] generated chars
        """
        idxs: List[int] = self.text_data.convert_str_to_idxs(starter_chars)
        idxs_out: List[int] = self.sequential_integer_model.generate_next_n_idxs_probabilistically(idxs, n=n, include_original_idxs=include_starter_chars)
        res = self.text_data.convert_idxs_to_str(idxs_out)
        return res


    # @staticmethod
    # def TEST_CharacterModelGenerator() -> None:
    #     neitszche_data: bk_data_sets.TextData = m_data.get_data_memoized()
    #
    #     net = m_model.get_model()
    #     model_weight_memoizer = ModelWeightsMemoizer(net, setup.PM)
    #     model_weight_memoizer.load_weights_into_net()
    #
    #     predictor = CharacterModelGenerator(net, neitszche_data, min_x_to_use=net.num_x_to_use)
    #
    #     start_text = "What if god is a woman?"
    #     # probs = predictor.get_next_probs(start_text)
    #     probs = predictor.generate_next_n_most_likely(start_text, n=100)
    #     probs
    #     probs = predictor.generate_next_n_probabilistically(start_text, n=100)
    #     probs
    #
    #     print(f"PC:KEYfchv:  TEST_CharacterModelGenerator done")
    #     exit(1)


if __name__ == '__main__':
    for test in tests_to_run:
        test()
