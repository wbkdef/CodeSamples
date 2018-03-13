"""
from bk_ds_libs import bk_utils_ds

done Lint-C
    # __c Bottom up!

"""

# __t Environments to use
# Import * plus import the module, for convenience, and to get tooltip help within each module
import math

import typing as tp
from typing import Any

from bk_ds_libs import bk_data_sets
import numpy as np
import pandas as pd
# import sklearn as sk
from sklearn.model_selection import train_test_split


# from bk_envs.env_general import *
# from bk_envs import env_general

# __t ---- STANDARD HEADER FINISHED ----


# noinspection PyStatementEffect
# pylint: disable=invalid-name
Ind = tp.Union[pd.Index, np.ndarray]


# noinspection PyPep8Naming
class Indexes:
    """This helps you keep straight your train, valid, test inds, and get random samples there-from for training batches"""
    def __init__(self,
                 ind_train: Ind,
                 ind_val: Ind,
                 ind_val2: Ind = None,
                 ind_val3: Ind = None,
                 ind_test: Ind = None) -> None:
        super().__init__()
        self.train = ind_train
        self.val = ind_val
        self.val2 = pd.Index([]) if ind_val2 is None else ind_val2
        self.val3 = pd.Index([]) if ind_val3 is None else ind_val3
        self.test = pd.Index([]) if ind_test is None else ind_test
        self.assert_mutually_exclusive()

    def assert_mutually_exclusive(self)  ->  None:
        """ Verifies, with assertion statements, that all the indices are mutually exclusive """
        inds = [self.train, self.val, self.val2, self.val3, self.test]
        inds = [ind for ind in inds if ind is not None]
        assert len(inds) >= 2
        comparisons_made = 0
        # pylint: disable=consider-using-enumerate
        for i in range(len(inds)):
            for j in range(i+1, len(inds)):
                comparisons_made += 1
                intersection = np.intersect1d(inds[i], inds[j])
                assert len(intersection) == 0, "index overlap detected"
        assert comparisons_made == (len(inds)**2-len(inds))/2, f"unexpected number of comparisons_made: {comparisons_made}"
        print(f"assert_mutually_exclusive succeeded for {comparisons_made} comparisons")

    @property
    def val_all(self)  ->  Ind:
        """ :return: All validation indices """
        return np.concatenate([self.val, self.val2, self.val3])

    @property
    def train_val_all(self)  ->  Ind:
        """ :return: All train and validation indices """
        to_concat = [self.train, self.val, self.val2, self.val3];  to_concat
        concatenated = np.concatenate(to_concat);  concatenated
        return concatenated

    # These get random indices (to then use to get items)
    @staticmethod
    def _get_rand_inds(batch_size, inside_inds)  ->  Ind:
        """Returns random indices from within the given inds."""
        sel = np.random.choice(inside_inds, batch_size, False)
        return sel
    def get_rand_inds_train(self, batch_size)    ->  Ind: return self._get_rand_inds(batch_size, self.train)
    def get_rand_inds_test(self, batch_size)     ->  Ind: return self._get_rand_inds(batch_size, self.test)
    def get_rand_inds_valid(self, batch_size)    ->  Ind: return self._get_rand_inds(batch_size, self.val)
    def get_rand_inds_valid2(self, batch_size)  ->  Ind: return self._get_rand_inds(batch_size, self.val2)
    def get_rand_inds_valid3(self, batch_size)  ->  Ind: return self._get_rand_inds(batch_size, self.val3)

    @staticmethod
    def deprecated_split_train_valid(inds_train_valid: Ind, frac_valid: float)  ->  "Indexes":
        """DEPRECATED Creates Indexes instance, automatically splitting off the given fraction to the validation set."""
        is_train: np.ndarray = np.random.rand(len(inds_train_valid)) > frac_valid
        inds_train: Ind = inds_train_valid[is_train]
        inds_valid: Ind = inds_train_valid[~is_train]
        inds = Indexes(inds_train, inds_valid)
        return inds

    @staticmethod
    def from_train_valid_to_split(inds_train_valid: Ind, frac_valid1: float, frac_valid2: float = 0, frac_valid3: float = 0, inds_test: Ind = None)  ->  "Indexes":
        """Creates Indexes instance, automatically splitting off the given fractions to the various validation sets."""
        rands = np.random.rand(len(inds_train_valid))
        thresholds = [0, frac_valid1, frac_valid1+frac_valid2, frac_valid1+frac_valid2+frac_valid3, 1]
        inds_valid1 = inds_train_valid[(thresholds[0] <= rands) & (rands < thresholds[1])]
        inds_valid2 = inds_train_valid[(thresholds[1] <= rands) & (rands < thresholds[2])]
        inds_valid3 = inds_train_valid[(thresholds[2] <= rands) & (rands < thresholds[3])]
        inds_train  = inds_train_valid[(thresholds[3] <= rands) & (rands < thresholds[4])]
        # len(inds_valid1), len(inds_valid2), len(inds_valid3), len(inds_train)
        assert set(inds_train_valid) == set(inds_valid1) | set(inds_valid2) | set(inds_valid3) | set(inds_train)

        inds = Indexes(inds_train, ind_val=inds_valid1, ind_val2=inds_valid2, ind_val3=inds_valid3, ind_test=inds_test)
        return inds

    def __repr__(self):
        return f"train:{len(self.train)}, valid:{len(self.val)}, val2:{len(self.val2)}, val3:{len(self.val3)}, test:{len(self.test)}"

    @staticmethod
    def TEST_from_train_valid_to_split():
        """Tests this classes method from_train_valid_to_split"""
        inds = Indexes.from_train_valid_to_split(inds_train_valid=np.arange(100100), frac_valid1=.2, frac_valid2=.1, inds_test=np.arange(100100, 150100))
        inds

    @staticmethod
    def TEST_Indexes_multiple_val():
        """Tests this class when have multiple validation sets"""
        # With numpy arrays, happy path, so can see
        ind_train = np.arange(0, 10)
        ind_val1 = np.arange(10, 15)
        ind_val2 = np.arange(15, 18)
        ind_val3 = np.arange(18, 20)
        ind_test = np.arange(20, 25)
        inds = Indexes(ind_train, ind_val1, ind_val2=ind_val2, ind_val3=ind_val3, ind_test=ind_test)
        assert set(inds.train_val_all) == set(np.arange(0, 20))
        # assert set(inds.val_all) == set(np.arange(0, 20))# __c SHOULD throw error!
        assert set(inds.val_all) == set(np.arange(10, 20))

        df = bk_data_sets.DataSetTitanic.get_combined()
        df = df.reset_index()
        ind_test = df[df.Survived.isnull()].index
        ind_remaining = df[~df.Survived.isnull()].index
        inds_train_val = ind_remaining
        ind_remaining, ind_val1 = train_test_split(ind_remaining, test_size=.25)
        ind_remaining, ind_val2 = train_test_split(ind_remaining, test_size=.20)
        ind_remaining, ind_val3 = train_test_split(ind_remaining, test_size=.15)
        ind_train = ind_remaining

        ind_test
        ind_val1
        ind_val2
        ind_val3
        ind_train

        inds = Indexes(ind_train, ind_val1, ind_val2=ind_val2, ind_val3=ind_val3, ind_test=ind_test)
        assert set(inds.train_val_all) == set(inds_train_val)

        # With numpy arrays, unhappy path, so can see that get assertion error!
        ind_train = np.arange(0, 10)
        ind_val1 = np.arange(10, 15)
        ind_val2 = np.arange(13, 18)
        ind_val3 = np.arange(18, 20)
        ind_test = np.arange(20, 25)
        # __c The next line should cause an assertion error:
        inds = Indexes(ind_train, ind_val1, ind_val2=ind_val2, ind_val3=ind_val3, ind_test=ind_test)
        inds

    @staticmethod
    def TEST_Indexes():
        """Tests this class"""
        df = bk_data_sets.DataSetTitanic.get_combined()
        df = df.reset_index()
        ind_test = df[df.Survived.isnull()].index
        ind_test

        dt, dv = train_test_split(df[~df.Survived.isnull()], test_size=.25)
        ind_train = dt.index
        ind_val = dv.index
        inds = Indexes(ind_train, ind_val, ind_test)

        rand_inds = inds.get_rand_inds_train(10)
        assert len(rand_inds) == 10
        assert set(rand_inds) < set(ind_train)
        # assert set(rand_inds) & set(ind_train) == set()
        assert set(rand_inds) & set(ind_test) == set()
        assert set(rand_inds) & set(ind_val) == set()

        rand_inds = inds.get_rand_inds_test(11)
        assert len(rand_inds) == 11
        assert set(rand_inds) < set(ind_test)
        assert set(rand_inds) & set(ind_train) == set()
        # assert set(rand_inds) & set(ind_test) == set()
        assert set(rand_inds) & set(ind_val) == set()

        rand_inds = inds.get_rand_inds_valid(12)
        assert len(rand_inds) == 12
        assert set(rand_inds) < set(ind_val)
        assert set(rand_inds) & set(ind_train) == set()
        assert set(rand_inds) & set(ind_test) == set()
        rand_inds
# Indexes.TEST_Indexes_multiple_val()
# Indexes.TEST_Indexes()
# Indexes.TEST_from_train_valid_to_split()


# noinspection PyStatementEffect
class CatVar:
    """Processes a pd.Series of categorical data into numerical values

    Missing and infrequent classes are mapped to 0, plus a few randomly chosen items if there aren't enough of the others to satisfy the own_group_threshold.

    >>> cv = CatVar(stv, inds, 10)
    >>> res = cv.s_out
    """
    def __init__(self,
                 s: pd.Series,
                 inds: Indexes,
                 own_group_threshold: tp.Union[float, int] = 20) -> None:
        """

        todo implement optionally doing thresholding based on train and validation data, not just train data.

        :param s: pd.Series to map to contiguous integers (for embedding later)
        :param inds:
        :param own_group_threshold: Make sure there is at least this many (or this fraction) of each category and of null items (so will learn to handle missing data)
            If a category has fewer items, map it to null
            If there are fewer than this many nulls, randomly map sufficient additional items to null
        """
        super().__init__()
        # todo convert this to an int if it is a float
        if 0 <= own_group_threshold < 1:
            own_group_threshold = int(len(inds.train) * own_group_threshold)
        else:
            assert isinstance(own_group_threshold, int)
        self.own_group_threshold: int = own_group_threshold
        self.inds: Indexes = inds
        # self.ind_test: pd.Index = pd.Index([]) if ind_test is None else ind_test
        # self.ind_val: pd.Index = ind_val
        # self.inds.train: pd.Index = inds.train
        self.s_in: pd.Series = s.copy()
        self._s_out: pd.Series = s.copy()  # __c The 'output'
        self.process()

    def process(self):
        """ Do all the processing and normalizing """
        vcs = self.s_in[self.inds.train].value_counts()
        vcs  # Use value counts for the training data
        replacements: dict = {x: i + 1 for i, x in enumerate(vcs.index) if vcs[x] >= self.own_group_threshold}
        self._s_out.replace(replacements, inplace=True)
        self._s_out[~self.s_in.isin(replacements.keys())] = 0
        # noinspection PyUnresolvedReferences
        num_0s_to_make = max(0, self.own_group_threshold - (self._s_out[self.inds.train] == 0).sum())
        inds_to_select_to_make_0s = self._s_out[self._s_out != 0].index.intersection(self.inds.train)
        inds_to_make_0s = np.random.choice(inds_to_select_to_make_0s, num_0s_to_make, False)
        self._s_out[inds_to_make_0s] = 0
        # self._s_out.value_counts(), self._s_out[self.inds.train].value_counts(), self._s_out[self.inds.val].value_counts(), self._s_out[self.ind_test].value_counts()

    @property
    def s_out(self) -> pd.Series:
        """Get the processed pd.Series"""
        # noinspection PyTypeChecker
        return self._s_out.copy()

    def __repr__(self) -> str:
        vcs = self.s_out.value_counts()
        num0s: int = vcs[0]
        num_cats: int = len(vcs)
        largest: int = vcs.iloc[0]
        smallest: int = vcs.drop(0).iloc[-1]

        s = "CatVar:\n"
        s += f"\tnum0s: {num0s}\n"
        s += f"\tnum_cats: {num_cats}\n"
        s += f"\tlargest: {largest}\n"
        s += f"\tsmallest: {smallest}\n"
        s += f"\tnew_cats: {vcs.index}\n"
        s
        return s

    def _get_debug_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df[self.s_in.name + "_IN"] = self.s_in
        df[self.s_in.name + "_OUT"] = self.s_out
        df["TrValTe"] = ""
        df.loc[self.inds.train, "TrValTe"] = "train"
        df.loc[self.inds.val, "TrValTe"] = "val"
        df.loc[self.inds.test, "TrValTe"] = "test"
        df
        return df

    # noinspection PyPep8Naming
    @staticmethod
    def TEST_CatVar():
        """Tests this class"""
        nums_to_abcs = dict(enumerate('abcdefghijklmnopqrstuvwxyz'))
        data = np.concatenate([np.random.randint(0, 10, 200), np.arange(26)])
        stv = pd.Series(data, name="cv_for_tests")
        stv
        stv = stv.map(nums_to_abcs)
        stv
        ind_test = np.arange(200, 226)
        ind_train, ind_valid = train_test_split(np.arange(0, 200), test_size=.25)
        ind_test, ind_train, ind_valid
        inds = Indexes(ind_train, ind_valid, ind_test)
        cat_var = CatVar(stv, inds, 2)
        assert cat_var.s_out[:210].value_counts()[0] == 2
        assert set(cat_var.s_out[:210].unique()) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        assert cat_var.s_out[210:].value_counts()[0] == 16
        # pylint: disable=protected-access
        cat_var._get_debug_df()
# CatVar.TEST_CatVar()


# noinspection PyStatementEffect,PyStatementEffect,PyStatementEffect
class ContVar:
    """Processes a pd.Series of continuous data into a pd.DataFrame

    Bins may be supplied for interpolated binning

    Missing items are mapped to 0, plus a few randomly chosen items if there aren't enough to satisfy the own_group_threshold.

    >>> cv = ContVar(stv, inds, own_group_threshold=10, bins=bins)
    >>> df = cv.df_out()
    """
    own_group_threshold: int

    def __init__(self,
                 s: pd.Series,
                 inds: Indexes,
                 own_group_threshold: tp.Union[float, int] = 20,
                 bins: tp.Sequence[tp.Union[float, int]] = None) -> None:
        """

        todo implement optionally doing thresholding based on train and validation data, not just train data.

        :param s: pd.Series to smoothed_bin.
        :param inds:
        :param own_group_threshold: Make sure there is at least this many or this fraction of null items (so will learn to handle missing data)
            If there are fewer than this many nulls, randomly map sufficient additional items to null
        :param bins: The bins ("fence posts") to do interpolated binning between.
        """
        super().__init__()
        # todo convert this to an int if it is a float
        self.bins = bins
        if 0 <= own_group_threshold < 1:
            self.own_group_threshold = int(len(inds.train) * own_group_threshold)
        else:
            assert isinstance(own_group_threshold, int)
            self.own_group_threshold: int = own_group_threshold
        self.inds: Indexes = inds
        self.s_in: pd.Series = s.copy()
        self._df_out: pd.DataFrame = None
        self.process()

    def process(self)  ->  None:
        """ Do all the processing and normalizing """
        s_proc: pd.Series = self.s_in.copy()

        # Make sure there are enough 'nulls' by creating more if needed (so through training, knows what to do for missing values on the test set).
        num_nulls_to_make = max(0, self.own_group_threshold - self.s_in.isnull().sum())
        # todo make question on this!
        inds_to_select_to_make_nulls = self.s_in[~self.s_in.isnull()].index.intersection(self.inds.train)
        inds_to_make_nulls = np.random.choice(inds_to_select_to_make_nulls, num_nulls_to_make, False)
        s_proc[inds_to_make_nulls] = None
        s_null = s_proc.isnull(); s_null  # __c s_null

        # Create the dataframe
        self._df_out = pd.DataFrame(s_proc)
        self._df_out[self.s_in.name + "_NA"] = s_null

        if self.bins is not None:
            df_binned = self.smoothed_bin(self.bins, s_proc)
            self._df_out = self._df_out.join(df_binned)

        # Normalize everything wrt. the training data
        mean = self._df_out.loc[self.inds.train].mean()
        self._df_out -= mean
        self._df_out.fillna(0, inplace=True)
        self._df_out.isnull().sum()
        std  = self._df_out.loc[self.inds.train].std()
        self._df_out /= std

        mean, std
        self._df_out

    @property
    def df_out(self)  ->  pd.DataFrame:
        """Get (a copy of) the resulting dataframe"""
        return self._df_out.copy()

    @staticmethod
    def smoothed_bin(bins: tp.Sequence[tp.Union[int, float]], series: pd.Series, include_orig=False)  ->  pd.DataFrame:
        """Smoothed bin a pandas series

        :param bins: bins ("fence posts") to interpolate between
        :param series: values to smoothed_bin
        :param include_orig: Include the original series values along with the smoothed_binned values
        :return: A pd.DataFrame of the smoothed_binned values, and optionally the original series

        Null values are assigned to a separate bin
        The bin values are like the "fence posts".
        Each value from the series is like some position along the fence.
        This function determines the two "fence posts" the position is in-between, and assigns a fraction of "1" to each fencepost proportional to how close it is.
        All other "fence posts" are assigned a value of 0

        For example, if
            bins = [0, 1, 5, 10]
            And we are considering val = 4 from the series.
        The value 4 lies between "fence posts" 1 and 5, and is 75% of the way to 5, so our smoothed_binned values we get:
            [0, .25, .75, 0]
        """
        name = series.name
        if "IPYTHON" == "Won't run by itself":
            series = range(60)
        arr = np.zeros([len(series), len(bins) + 1])
        bins = np.array(bins)
        titles = [f"{name}_{v}" for v in bins]+[f"{name}_nan"]
        for i, val in enumerate(series):
            if math.isnan(val):
                arr[i, -1] = 1
            else:
                bgv = (bins >= val)
                assert bgv.any()
                j = bgv.argmax()  # Will return position of first true
                interpolation_frac = (val-bins[j-1])/(bins[j]-bins[j-1])
                arr[i, j-1] = 1-interpolation_frac
                arr[i, j] = interpolation_frac
        df = pd.DataFrame(arr, columns=titles, index=series.index)  # __c Important to have this index in here for combining with others!
        if include_orig:
            df[f"{name}_orig"] = series
        return df

    def _get_debug_df(self)  ->  pd.DataFrame:
        df: pd.DataFrame = self.df_out
        df[self.s_in.name + "_IN"] = self.s_in
        df["TrValTe"] = ""
        df.loc[self.inds.train, "TrValTe"] = "train"
        df.loc[self.inds.val, "TrValTe"] = "val"
        df.loc[self.inds.test, "TrValTe"] = "test"
        df
        return df

    # noinspection PyPep8Naming
    @staticmethod
    def TEST_ContVar()  ->  None:
        """Tests this class

        # pylint: disable=unused-argument
        """
        name = "cont_var_for_tests"
        stv = pd.Series(np.arange(100), name=name)
        bins = [0, 5, 20, 100]
        stv
        ind_train = np.arange(50)
        ind_valid = np.arange(50, 75)
        ind_test = np.arange(75, 100)
        inds = Indexes(ind_train, ind_valid, ind_test)
        stv[stv[ind_train].sample(4).index] = None
        stv
        cont_var = ContVar(stv, inds, own_group_threshold=10, bins=bins)
        # pylint: disable=protected-access
        df = cont_var._get_debug_df()
        df
        num_train, num_valid, num_test = (df[name][ind_train] == 0).sum(), (df[name][ind_valid] == 0).sum(), (df[name][ind_test] == 0).sum()
        assert [10, 0, 0] == [num_train, num_valid, num_test]

        assert set(df[f"{name}_NA"].value_counts()) == {90, 10}
        # __c Eyeball these!
        df.loc[0:20, f"{name}_5"]
        df[:10]
        df[-10:]
# ContVar.TEST_ContVar()


class DataProcessor:
    """An abstract class for how a data processor would work

    # pylint: disable=missing-docstring
    """
    # todo Add in test data (and indices) as well
    def __init__(self, dtx, dty, dvx=None, dvy=None) -> None:
        super().__init__()
        self.dvy = dvy
        self.dvx = dvx
        self.dty = dty
        self.dtx = dtx

    # pylint: disable=invalid-name
    def get_processed_data(self, dx, dy)  ->  Any:
        """The pre-processing of validation and test data should ultimately depend on the training data, or possibly the train and val data to ensure feature consistency across folds.

        """
        raise NotImplementedError

    def get_processed_train_data(self):
        """ :return: Process the train data and return """
        return self.get_processed_data(self.dtx, self.dty)
    def get_processed_valid_data(self)  ->  Any:  # Tuple["validate_data_x", "validate_data_y"]
        """ :return: Process the validation data and return """
        if self.dvx is None or self.dvy is None:
            raise ValueError("get_processed_valid_data should only be called if dvx, dvy were initialized")
        return self.get_processed_data(self.dvx, self.dvy)


def df_dropout(df: pd.DataFrame, rate: float, num_duplicates=1,
               inplace=False, exclude_cols=None) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """ dropout some values (making them null)

    :param df: The pd.DataFrame whose values are to be randomly dropped out
    :param rate: The probability that each value drops out
    :param num_duplicates: If > 1, then create this many duplicates of the DataFrame before drop-out.  Requires inplace=False.
    :param inplace: If False, a copy of the DataFrame is used
    :param exclude_cols:  Specifies columns whose values will not be randomly dropped
    :return: Two dataframes:
        1) The dataframe, with some values dropped
        2) A boolean dataframe, where True indicates the value was dropped from the original dataframe
    """
    if "IPYTHON" == "Won't run by itself":
        rate = .2
        # df =  bk_data_sets.DataSetTitanic.get_train()
    exclude_cols = exclude_cols or [];  exclude_cols
    # recursive calls if num_duplicates > 1
    if num_duplicates > 1:
        assert inplace is False
        # noinspection PyUnusedLocal
        dfs, mask_records = zip(*[df_dropout(df, rate, 1, inplace, exclude_cols) for i in range(num_duplicates)])
        df = pd.concat(dfs)
        mask_record = pd.concat(mask_records)
        return df, mask_record

    if not inplace:
        df = df.copy()

    # make mask and set to None
    mask_record = {}
    for col in df:
        if col not in exclude_cols:
            msk = np.random.rand(len(df)) < rate;  msk[:2]
            df.loc[msk, col] = None
            mask_record[col] = msk
    mask_record = pd.DataFrame(mask_record);  mask_record[:2]
    mask_record[:2], df[:2]

    return df, mask_record

# if __name__ == '__main__':
#     df = bk_data_sets.DataSetTitanic.get_train()
#     df, masked_record = df_dropout(df, .4)
#     masked_record[:2]
#     df[:2]
#     masked_record.shape, df.shape
#     df, masked_record = df_dropout(df, .4, 5, exclude_cols=['PassengerId', 'Survived'])
#     masked_record[:2]
#     df[:2]
#     masked_record.shape, df.shape

# def df_dropout(df: pd.DataFrame, rate: float, num_duplicates=1, inplace=False, exclude_cols=None) -> pd.DataFrame:
#     exclude_cols = exclude_cols or []
#     exclude_cols
#     if num_duplicates > 1:
#         assert inplace is False
#         dfs = [df_dropout(df, rate, exclude_cols=exclude_cols) for i in range(num_duplicates)]
#         return pd.concat(dfs)
#     if not inplace:
#         df = df.copy()
#     if "IPYTHON" == "Won't run by itself":
#         [v for v in df]
#         rate = .2
#         col = 'Sex'
#         df = bk_data_sets.DataSetTitanic.get_train()
#     for col in df:
#         # msk = np.random.choice(len(df), int(len(df)*rate), replace=False); msk[:10]
#         msk = np.random.rand(len(df)) < rate; msk[:10] # Makes a boolean index - more robust than numerical, which depend's on the dataframe's index!
#         if col not in exclude_cols:
#             df.loc[msk, col] = None
#         # df[:2]
#     return df
