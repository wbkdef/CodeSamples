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
