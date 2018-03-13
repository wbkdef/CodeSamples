
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