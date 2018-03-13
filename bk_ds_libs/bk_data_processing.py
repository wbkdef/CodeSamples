import importlib
import os
import os.path as osp
import re
import sys
import typing as tp
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any
from urllib.request import urlretrieve

import numpy as np
import sklearn as sk
# from bk_ds_libs.bk_data_sets import DataSetNietzsche
# from sklearn.model_selection import train_test_split
import pandas as pd
#
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torch.optim

from bk_ds_libs import bk_utils_ds as utd
import bk_paths
from bk_general_libs import bk_itertools, bk_typing as tp_bk
from bk_general_libs import bk_decorators
import bk_general_libs.bk_typing as tp_bk

pd.options.display.max_rows = 50
pd.options.display.max_columns = 10
pd.options.display.max_columns = 50
pd.options.display.max_columns = 20

T = TypeVar('T')


def extract_x_ngram_and_y_iterator(idxs: Iterable[T], *, num_x_elements: int)  ->  Iterable[Tuple[Tuple[T, ...], T]]:
    xy: Iterable[Tuple[T, ...]] = bk_itertools.ngrams(idxs, n=num_x_elements+1)
    for xyi in xy:
        x = xyi[:num_x_elements]
        y = xyi[num_x_elements]
        assert len(xyi) == num_x_elements + 1;  assert y == xyi[-1];  assert x + (y, ) == xyi;  assert len(x) == num_x_elements
        yield x, y


def extract_x_ngrams_and_y_iterators(idxs: Iterable[T], *, num_x_elements: int)  ->  Tuple[Iterable[Tuple[T]], Iterable[T]]:
    xy_iter = extract_x_ngram_and_y_iterator(idxs, num_x_elements=num_x_elements)
    x, y = zip(*xy_iter)
    return x, y


def extract_x_ngrams_and_y_as_numpy(idxs: Iterable, *, num_x_elements: int)  ->  Tuple[np.ndarray, np.ndarray]:
    x, y = extract_x_ngrams_and_y_iterators(idxs, num_x_elements=num_x_elements)
    x = np.array(x)
    y = np.array(y)
    return x, y


def TEST_extract_ngrams_and_y_fcns():
    x, y = extract_x_ngrams_and_y_iterators(range(5), num_x_elements=3)
    lx, ly = list(x), list(y)
    lx, ly
    assert list(y) == [3, 4]
    assert list(x) == [(0, 1, 2), (1, 2, 3)]
    x, y = extract_x_ngrams_and_y_as_numpy(range(5), num_x_elements=3)
    assert (x == np.array([(0, 1, 2), (1, 2, 3)])).all()
    assert (y == np.array([3, 4])).all()
# if __name__ == '__main__':
#     TEST_extract_ngrams_and_y_fcns()

class TextData:
    def __init__(self, text: str, *, x_bigram_size: int, shorten_text_to_len: int = None) -> None:
        """

        # pylint: disable=unused-argument
        0 is left free
        :param text:
        :param x_bigram_size:
        :param shorten_text_to_len: Truncate 'text' length at shorten_text_to_len - after the character <==> integer mappings have been created
        """
        super().__init__()

        if shorten_text_to_len is not None:
            text = text[:shorten_text_to_len]
        self.text: str = text
        self.x_bigram_size: int = x_bigram_size

        self.chars_2_idxs: Dict[str, int]
        self.idxs_2_chars: Dict[int, str]
        self.idxs: Sequence[int]   # Each character in the text, mapped to an integer
        self.vocab_size: int

        self.x: np.ndarray
        self.y: np.ndarray

        self._setup_char_int_mappings()
        self.idxs = self.convert_chars_to_idxs(self.text)
        self.x, self.y = extract_x_ngrams_and_y_as_numpy(self.idxs, num_x_elements=self.x_bigram_size)

    def _setup_char_int_mappings(self)  ->  None:
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars) + 1
        chars.insert(0, "\0")  # __c What we'll map everything that doesn't have a good place to!
        self.chars_2_idxs = dict((c, i) for i, c in enumerate(chars))
        self.idxs_2_chars = dict((i, c) for c, i in self.chars_2_idxs.items())

    def convert_chars_to_idxs(self, chars: Iterable[str])  ->  List[int]:
        """Converts an iterable of characters (possibly just a string) to the corresponding list of index integers

        :param chars:  An iterable of characters (possibly just a string)
        :return:
        """
        res = []
        for char in chars:
            assert len(char) == 1
            res.append(self.chars_2_idxs[char])
        return res
        # return [self.chars_2_idxs[c] for c in chars]
    def convert_str_to_idxs(self, s: str)  ->  List[int]:  return self.convert_chars_to_idxs(s)
    def convert_char_to_idx(self, char: str)  ->  int:
        idxs = self.convert_chars_to_idxs(char)
        assert len(idxs) == 1
        return idxs[0]

    def convert_idxs_to_chars(self, idxs: tp_bk.SelfIterable_Recursive[int])  ->  tp_bk.SelfList_Recursive[str]:
        if isinstance(idxs, Iterable):
            return [self.convert_idxs_to_chars(idx) for idx in idxs]
        try:
            return self.idxs_2_chars[idxs]
        except KeyError:
            return f"[[NF:{idxs}]]"

        # res = []
        # for idx in idxs:
        #     res.append(self.convert_idxs_to_chars(idx))
            # if not isinstance(idx, int):
            #     res.append(self.convert_idxs_to_chars(idx))
            # else:
            #     res.append(self.idxs_2_chars[idx])
        # return res
    def convert_idxs_to_str(self, idxs: tp_bk.SelfIterable[int])  ->  str:
        chars = tp.cast(List[str],  self.convert_idxs_to_chars(idxs))
        return ''.join(chars)
    def convert_idx_to_char(self, idx: int)  ->  str:  return self.convert_idxs_to_str([idx])[0]

    def __repr__(self) -> str:
        # s = super().__repr__()
        from bk_general_libs import bk_strings

        header = "\n" + self.__class__.__name__

        import tabulate
        vals = tabulate.tabulate([
            ["text:", f'"{bk_strings.get_1_line_iterable_representation(self.text, items_at_start_and_end=30, items_sep="")}"'],
            # ["text:", f'"{self.text[:100]}"'],
            ["x_bigram_size:", self.x_bigram_size],
            ["vocab_size:", self.vocab_size],
            ["chars_2_idxs", bk_strings.get_1_line_dict_representation(self.chars_2_idxs)],
            ["idxs_2_chars", bk_strings.get_1_line_dict_representation(self.idxs_2_chars)],
            ["idxs", bk_strings.get_1_line_iterable_representation(self.idxs, items_at_start_and_end=7, items_sep=", ")],
            ["x[:5]", self.x[:5]],
            ["y[:5]", self.y[:5]]
        ])

        # print(vals)
        import textwrap
        res = header + textwrap.indent(f"\n{vals}", "    ")
        # print(res)
        return res

    # @staticmethod
    # def from_Nietzsche(x_bigram_size: int, shorten_text_to_len: int = None):

    @staticmethod
    def TEST_TextData():
        s = "abcaba"
        text_data = TextData(s, x_bigram_size=3)
        print(f"\ntext_data is [[{text_data}]]")
        assert text_data.convert_idxs_to_str([1, 2, 2, 1]) == 'abba'
        assert text_data.convert_idx_to_char(3) == 'c'
        assert text_data.convert_idxs_to_chars([3, 1, [2, 1]]) == ['c', 'a', ['b', 'a']]
        assert text_data.convert_chars_to_idxs('cabba') == [3, 1, 2, 2, 1]
        assert text_data.convert_char_to_idx('b') == 2

        # todo Implement with some dummy data
        s = "This is some text to process. 2nd time: This is some text to process.  3rd time: This is some text to process."
        text_data = TextData(s, x_bigram_size=3)
        print(text_data)
        print(f"PC:KEYqMoW:  TEST_TextData PASSED")

        return # __c This takes a while - just for looking at in IPython
        # todo Use NietszcheData
        # s = DataSetNietzsche.get_raw()
        # text_data = TextData(s, x_bigram_size=3)
        # print(f"\ntext_data is [[{text_data}]]")
        # text_data
if __name__ == '__main__':
    TextData.TEST_TextData()


class BColzWrapper:
    """Functions for saving/loading arrays to bcolz format"""
    @staticmethod
    def save_array(rootdir_name: str, arr: np.ndarray):
        """Save array to bcolz format"""
        import bcolz
        # print(f"dirname is {dirname}, \n arr is {arr}")
        arr = bcolz.carray(arr, rootdir=rootdir_name, mode='w')
        arr.flush()

    @staticmethod
    def load_array(dirname: str):
        """Load array from bcolz format"""
        import bcolz
        arr = bcolz.open(rootdir=dirname, mode='r')
        return arr[:]  # convert back to numpy array


def onehot_encode_1d_array(array_1d: np.ndarray, sparse=False):
    """one-hot encodes a 1-dimensional array

    :param array_1d: A 1 dimensional array to one-hot encode
    :param sparse: Whether to return a sparse array.
    :return: a 2-dimensional one-hot encoding of array_1d
    """
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=sparse)
    return ohe.fit_transform(array_1d.reshape((-1, 1)))


def add_datepart(df, fldname, drop=True):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    COPIED FROM FAST.AI LIBRARY

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.

    Examples:
    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df

        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13

    >>> add_datepart(df, 'A')
    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    # pylint: disable=invalid-name
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
              'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt, n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)