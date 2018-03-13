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

# from bk_ds_libs import py_torch_helpers as pth
# from bk_ds_libs import bk_utils_ds as utd
# from bk_ds_libs import bk_data_sets

import bk_general_libs.bk_typing as tp_bk
from bk_general_libs.bk_typing import SelfSequence, SelfList, SelfTuple, SelfIterable, SelfList_Recursive, SelfSequence_Recursive, SelfIterable_Recursive, NonNegInt, NonNegFloat, Probability, NNI, NNF, PBT, TV
from bk_general_libs import bk_itertools
from bk_general_libs import bk_decorators
from bk_general_libs import bk_strings



class EvalResults(tp.NamedTuple):
    preds: Variable
    raw_loss: Variable
    other_model_outputs: Tuple[Variable]

class Stepper:
    """ Runs calculations and (for training) updates a model on a single batch of data """
    net: nn.Module
    criterion: Callable
    optimizer: torch.optim.Optimizer
    def __init__(self, m: nn.Module, *, opt: torch.optim.Optimizer, criterion: Callable) -> None:
        super().__init__()
        self.m = m
        self.opt = opt
        self.criterion = criterion

    def step_train(self, xs: Iterable[Variable], y: Variable)  ->  float:
        """ Takes a batch of x/y values.  1) Updates weights and 2) return's the models (raw) loss as a float."""
        # xtra = []
        # output = self.m(*xs)
        # if isinstance(output,(tuple,list)): output, *xtra = output
        # loss = raw_loss = self.criterion(output, y)

        raw_loss, eval_res = self.evaluate(xs, y)
        self._update_weights(eval_res)
        return raw_loss

    def _update_weights(self, eval_res: EvalResults)  ->  None:
        self.opt.zero_grad()
        loss = eval_res.raw_loss
        # __c If have other outputs, may want to include this line again to implement more complex optimization fcns!
        # if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        # if self.clip:   # Gradient clipping
        #     nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()

    def evaluate(self, xs: Iterable[Variable], y: Variable)  ->  Tuple[float, EvalResults]:
        """ Takes a batch of x/y values.  Returns:

        1) the models (raw) loss as a float,
        2) NamedTuple of: Predictions, Variable(loss), and other model outputs.

        No weight update
        """
        output = self.m(*xs)
        xtra = []
        if isinstance(output, (tuple, list)): output, *xtra = output
        raw_loss = self.criterion(output, y)
        return raw_loss.data[0], EvalResults(preds=output, raw_loss=loss, other_model_outputs=xtra)

    def step_valid(self, xs: Iterable[Variable], y: Variable)  ->  float:
        """ Takes a batch of x/y values.  Return's the models (raw) loss as a float.  No weight update

        Call 'evaluate' to also get the predictions, and other model outputs
        """
        raw_loss, eval_res = self.evaluate(xs, y)
        return raw_loss
        pass


class EpochRunner:
    def __init__(self, net: nn.Module, criterion, optimizer) -> None:
        super().__init__()
        self.net: nn.Module = net
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self):
        """ Does train and valid epochs """
        pass

    def do_epoch(self, data_loader: data.DataLoader, source_dropout_max_rate: float, update_weights: bool, num_batches_between_print_updates: tp.Optional[int], print_prefix=""):
        """Do one epoch through the data_loader data

        :param print_prefix: Prefixed to printed output
        :param data_loader:
        :param source_dropout_max_rate:  a float in the range [0, 1)
        :param update_weights:  Whether to update the weights (i.e. when training) or not (i.e. cross-validation, test)
        :param num_batches_between_print_updates:
        :return:
        """
        epoch_loss = torch.zeros(1).cuda()
        num_processed = 0
        for i, batch in enumerate(data_loader):
            x: Variable; y: Variable
            x, y = pth.V([batch['cats'], batch['y' ]])
            # x, y = pth.V([x, y])
            # x: Variable; y: Variable
            assert torch.cuda.LongTensor == type(x.data) == type(y.data)

            batch
            x[:10], y[:10]

            # import random
            # source_dropout_rate = random.uniform(0, source_dropout_max_rate)

            # Forward + Backward + Optimize
            self.optimizer.zero_grad()  # zero the gradient buffer
            outputs = self.net(x)
            loss = self.criterion(outputs, y)
            if update_weights:
                loss.backward()
                self.optimizer.step()

            num_processed += len(y)
            epoch_loss += loss.data * len(y)
            if num_batches_between_print_updates is not None and (i + 1) % num_batches_between_print_updates == 0:
                print(print_prefix + 'Step [%d/%d], Loss: %.4f' %
                      (i+1,  len(data_loader), epoch_loss[0]/num_processed))

        epoch_loss = epoch_loss[0] / num_processed
        # print(print_prefix + 'Loss at end of epoch: %.4f on %d example' % (epoch_loss, len(data_loader))

        return epoch_loss

    def do_epoch_train(self, data_loader: data.DataLoader, source_dropout_max_rate: float, num_batches_between_print_updates: tp.Optional[int]):
        self.net.train()
        epoch_loss = self.do_epoch(data_loader, source_dropout_max_rate=source_dropout_max_rate, update_weights=True,
                      num_batches_between_print_updates=num_batches_between_print_updates, print_prefix="Train: ")
        return epoch_loss

    def do_epoch_valid(self, data_loader: data.DataLoader, source_dropout_max_rate: float = 0, num_batches_between_print_updates: tp.Optional[int] = None):
        self.net.eval()
        epoch_loss = self.do_epoch(data_loader, source_dropout_max_rate=source_dropout_max_rate, update_weights=False, num_batches_between_print_updates=num_batches_between_print_updates, print_prefix="Valid: ")
        return epoch_loss