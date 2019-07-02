"""

# todo_weekly:  Pylint-C
"""
import os
import os.path as osp
from os import path as osp
from typing import Tuple, Dict, Sequence, Iterable
from urllib.request import urlretrieve

from bk_ds_libs.bk_data_processing import T, TextData
import pandas as pd

from tqdm import tqdm

import bk_paths
from bk_general_libs import bk_itertools
from bk_general_libs import bk_decorators

pd.options.display.max_rows = 50
pd.options.display.max_columns = 10
pd.options.display.max_columns = 50
pd.options.display.max_columns = 20



# __t Main

# ______t Functions to easily retrieve datasets from disk


class TqdmUpTo(tqdm):
    """TqdmUpTo helper"""
    # pylint: disable=invalid-name
    def update_to(self, b=1, bsize=1, tsize=None):
        """

        :param b:
        :param bsize:
        :param tsize:
        """
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_from_url(url, file_path):
    """

    :param url:
    :param file_path:
    """
    if not os.path.exists(file_path):
        # pylint: disable=invalid-name
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, file_path, reporthook=t.update_to)

def get_data_path(sub_dir, file_name=None) -> str:
    """Get path to sub_dir/file_name inside the data directory"""
    path = osp.join(bk_paths.ddd_dir_datasets, sub_dir)
    if file_name is not None:
        path = osp.join(path, file_name)
    return path

def get_data_from_url(url, sub_dir, file_name, exist_ok=False):
    """Downloads data from the given url to the subdir/file_name of the data folder

    :param url: url
    :param sub_dir: subdirectory of data folder
    :param file_name: filename to name file
    :param exist_ok: If true, then will do nothing if the file already exists.  Otherwise throws an assertion error.
    :return: path to file downloaded
    """
    os.makedirs(get_data_path(sub_dir), exist_ok=True)
    file_path = get_data_path(sub_dir, file_name)
    if not osp.exists(file_path):
        get_from_url(url, file_path)
        print(f"PC:KEYlaFw:  Downloading {url} to {file_path}")
    else:
        assert exist_ok is True
        print(f"PC:KEYsjLe:  File already exists, not downloading {file_path}")
    return file_path



# ______t Individual datasets
class DataSetTitanic:
    @staticmethod
    def get_train()  ->  pd.DataFrame:
        """Get titanic train data from csv"""
        df = pd.read_csv(bk_paths.fdtr_titanic_train_csv)
        return df
    @staticmethod
    def get_test()  ->  pd.DataFrame:
        """Get titanic test data from csv"""
        df = pd.read_csv(bk_paths.fdte_titanic_test_csv)
        return df
    @staticmethod
    def get_combined()  ->  pd.DataFrame:
        """Get titanic train and test data, concatenated into a single dataframe"""
        df: pd.DataFrame = pd.concat([DataSetTitanic.get_train(), DataSetTitanic.get_test()])
        df = df.reset_index()
        return df
    @staticmethod
    def get_train_sample_processed(rows=10) -> pd.DataFrame:
        """Get a small sample of the titanic data, with columns 'Name Ticket' dropped"""
        df = DataSetTitanic.get_train()[:rows].copy()
        df.drop('Name Ticket'.split(), 1, inplace=True)
        return df


class DataSetMovieLens:
    @staticmethod
    def get_ratings(drop_timestamp=True)  ->  pd.DataFrame:
        """ Get a pd.DataFrame of movie ids, user ids and the user's rating of the movie (and a time stamp)

                userId  movieId  rating
        0            1       31     2.5
        1            1     1029     3.0
        """
        file_path = osp.join(bk_paths.ddd_dir_datasets, "ml-latest-small", "ratings.csv")
        df = pd.read_csv(file_path)
        if drop_timestamp:  df.drop('timestamp', inplace=True, axis=1)
        return df
    @staticmethod
    def get_names()  ->  pd.DataFrame:
        """ Get a pd.DataFrame of movie ids and their names

              movieId               title                                        genres
        0           1    Toy Story (1995)   Adventure|Animation|Children|Comedy|Fantasy
        """
        file_path = osp.join(bk_paths.ddd_dir_datasets, "ml-latest-small", "movies.csv")
        df = pd.read_csv(file_path)
        return df
    @staticmethod
    def TEST_movie_lens_data():
        """ Tests related functions """
        movie_ratings = DataSetMovieLens.get_ratings()
        movie_names = DataSetMovieLens.get_names()
        print(f"\nmovie_ratings is [[{movie_ratings}]]")
        print(f"\nmovie_names is [[{movie_names}]]")
        movie_ratings
        movie_names


# ______t NietszcheData
class DataSetNietzsche:
    @staticmethod
    def get_raw()  ->  str:
        """ Get the nietzsche text """
        file_path = get_data_from_url("https://s3.amazonaws.com/text-datasets/nietzsche.txt", 'nietzsche', f'nietzsche.txt', exist_ok=True)
        text = open(file_path).read()
        print('nietzsche corpus length:', len(text))
        return text
    @staticmethod
    @bk_decorators.deprecated
    def get_processed_to_idxs(subset_size: int = None):
        """Much of this text processing has been moved into class 'TextData'

        :param subset_size:
        :return:
        """
        text = DataSetNietzsche.get_raw()
        print(f"\ntext[:400] is [[{text[:400]}]]")
        chars = sorted(list(set(text)))
        vocab_size = len(chars) + 1
        print('total chars:', vocab_size)
        chars.insert(0, "\0")
        ''.join(chars[1:-6])  # __i IPYTHON
        chars_2_idxs = dict((c, i) for i, c in enumerate(chars))
        idxs_2_chars = dict((i, c) for i, c in enumerate(chars))
        if subset_size is not None:
            text = text[:subset_size]
        idxs = [chars_2_idxs[c] for c in text]
        return idxs, chars_2_idxs, idxs_2_chars, vocab_size
    @staticmethod
    def TEST_get_raw()  ->  None:
        """Tests related functions"""
        nietzsche = DataSetNietzsche.get_raw()
        nietzsche[:400]
        print(nietzsche[:400])




# Delete this region now?
# region Deletion Pending
# class NietszcheData(TextData):
#     def __init__(self, *, x_bigram_size: int, shorten_text_to_len: int = None) -> None:
#         text = DataSetNietzsche.get_raw()
#         super().__init__(text, x_bigram_size=x_bigram_size, shorten_text_to_len=shorten_text_to_len)
@bk_decorators.deprecated
def extract_x_ngrams_and_y(idxs: Iterable[T], *, num_x_elements: int)  ->  Iterable[Tuple[Tuple[T, ...], T]]:
    xy: Iterable[Tuple[T, ...]] = bk_itertools.ngrams(idxs, n=num_x_elements + 1)
    for xyi in xy:
        x: Tuple[T, ...] = xyi[:num_x_elements]
        y: T = xyi[num_x_elements]
        assert len(xyi) == num_x_elements + 1;  assert y == xyi[-1];  assert x + (y, ) == xyi
        yield x, y
    # x, y = zip()
    # xy_arr = np.array(xy)
    # x: np.ndarray = xy_arr[:-1]
    # y: np.ndarray = xy_arr[-1]
    # assert x.shape[1] == num_x_elements
    # assert len(x) == len(y)

    # x: Sequence[Sequence] = np.array([idxs[i:i + num_x_elements] for i in range(len(idxs) - num_x_elements)])
    # y: Sequence = np.array([idxs[i] for i in range(num_x_elements, len(idxs))])
    # return x, y
class NietszcheDataOLD:
    def __init__(self, n_x_chars, subset_size: int = None) -> None:
        super().__init__()

        self.n_x_chars: int = n_x_chars

        idxs, chars_2_idxs, idxs_2_chars, vocab_size = DataSetNietzsche.get_processed_to_idxs(subset_size)
        self.idxs: Sequence = idxs
        self.chars_2_idxs: Dict[str, int] = chars_2_idxs
        self.idxs_2_chars: Dict[int, str] = idxs_2_chars
        self.vocab_size: int = vocab_size
        x, y = extract_x_ngrams_and_y(idxs, num_x_elements=self.n_x_chars)
        self.x: Sequence[Sequence] = x
        self.y: Sequence = y
@bk_decorators.deprecated
def NietszcheData(n_x_chars, subset_size: int = None)  ->  NietszcheDataOLD:
    """Best to now use class "TextData" """
    return NietszcheDataOLD(n_x_chars, subset_size)
# endregion
class FilesWrapper:
    """Functions for downloading files"""
    # pylint: disable=invalid-name
    @staticmethod
    def download_file_if_not_already_downloaded(url: str, file_path: str):
        """Wraps wget module"""
        dir_download_to = osp.dirname(file_path)
        os.makedirs(dir_download_to, exist_ok=True)
        if osp.isfile(file_path):
            print("File already exists, doing nothing")
        else:
            print("File doesn't exist, downloading")
            from downloaded_libs import wget  # type: ignore
            wget.download(url, file_path)