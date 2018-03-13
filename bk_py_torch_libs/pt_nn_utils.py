
import os
import os.path as osp
import re
import sys
import enum
import typing as tp
import itertools as it
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable, Sized, NamedTuple, Optional
from functools import partial

import bk_ds_libs.bk_data_processing
import bk_general_libs.bk_caching
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