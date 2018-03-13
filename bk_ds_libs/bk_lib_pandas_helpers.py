import typing as tp
from typing import List, Optional, Union, Callable, Iterable
import re
import importlib
# importlib.reload(bk_lib)

import numpy as np
import pandas as pd
import seaborn as sns

# from bk_libs import bk_lib

pd.set_option('display.max_rows', 99)
pd.set_option('display.max_rows', 9)

pd.set_option('display.max_columns', 77)
pd.set_option('display.max_columns', 7)

# __t Setup Constants => _ANS, _PLOT
_ANS = True
_PLOT = False

# __t Setup Variables => col_id, col_pred, DECLARED: rows_train, rows_test
# region Setup Variables
col_id = "ID_COL"
col_pred = "COL_TO_PREDICT"
rows_train = None
rows_test = None
# endregion

# # __t Load the data  =>  rows_train, rows_test
# # region Load the data
# df_train = pd.read_csv(r"")
# df_test = pd.read_csv(r"")
# df: pd.DataFrame = pd.concat([df_train, df_test])
# df
# df.set_index(col_id, inplace=True)
# df
# rows_train = df[col_pred].notnull()
# rows_test = df[col_pred].isnull()
# # endregion

# __t ---- STANDARD HEADER FINISHED ----


# class PDSeriesWithAttr(pd.Series):
#     def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)  ->  None:
#         pass
#     # def apply(fcn: Callable) -> pd.Series:
#     def apply(self, fcn: Callable) -> 'PDSeriesWithAttr': None
#     def value_counts(self) -> 'PDSeriesWithAttr': None


# class DFTypedAbstract:
#     @property
#     def shape(self) -> (int, int):
#         return (1, 1)


def load_df_from_str(s, sep=","):
    """Loads a dataframe from a string, as though from csv

    See commented usage below.
    Based on https://stackoverflow.com/questions/22604564/how-to-create-a-pandas-dataframe-from-string
    """
    from io import StringIO
    test_data = StringIO(s)
    df_s = pd.read_csv(test_data, sep=sep)
    df_s
    return df_s

# s = """col1;col2;col3
# 1;4.4;99
# 2;4.5;200
# 3;4.7;65
# 4;3.2;140
# """
# load_df_from_str(s, ";")

def parse_df_from_str(s: str, re_split="\s+", sep_to_insert_before_loading=",", debug_output=False) -> pd.DataFrame:
    """Does some string processing to extract the df out of a string"""
    # s = """        one       two     three four   five
    #     a -0.166778  0.501113 -0.355322  bar  False
    #     c -0.337890  0.580967  0.983801  bar  False
    #     e  0.057802  0.761948 -0.712964  bar   True
    #     f -0.443160 -0.974602  1.047704  bar  False
    #     h -0.717852 -1.053898 -0.019369  bar  False"""
    # re_split = "\s+"
    # sep_to_insert_before_loading = ","
    if debug_output:
        print(f"\ns is [[{s}]]")
        print(f"\nre_split is [[{re_split}]]")
    import re

    # __t split lines, throw away empty ones at start and finish
    lines = s.splitlines()
    if re.match(r"^\s*$", lines[0]):
        lines = lines[1:]
    if re.match(r"^\s*$", lines[-1]):
        lines = lines[:-1]
    lines = [re.sub(r"(^\s+|\s+$)", "", x) for x in lines]
    # print(lines)

    # __t split into columns. If the first (title) row has one less column,
    # __t add it back with 'index'
    # hs = re.split(re_split, line)
    ls = [re.split(re_split, x) for x in lines]
    if debug_output:
        print(f"ls is [[{ls}]]")
    hs = ls[0]
    rss = ls[1:]
    rs_lens = list(set([len(x) for x in rss]))
    assert len(rs_lens) == 1
    num_cols = rs_lens[0]
    if len(hs) == num_cols-1:
        hs.insert(0, "index")
    assert len(hs) == num_cols

    # __t Join together, parse it from csv, and set the index if applicable
    ls = rss #All lines, split
    ls.insert(0, hs)
    if debug_output:
        print(f"ls is [[{ls}]]")
    lc = [sep_to_insert_before_loading.join(x) for x in ls]
    lc
    sc = "\n".join(lc)
    sc
    df: pd.DataFrame = load_df_from_str(sc, sep=",")
    if re.match('(?i)^index$', hs[0]):
        df.set_index(hs[0], inplace=True)
    df
    return df



if __name__ == '__main__':
    print("starting run")
    s = """
                one       two     three four   five
            a -0.166778  0.501113 -0.355322  bar  False
            c -0.337890  0.580967  0.983801  bar  False
            e  0.057802  0.761948 -0.712964  bar   True
            f -0.443160 -0.974602  1.047704  bar  False
            h -0.717852 -1.053898 -0.019369  bar  False
        """
    dfp = parse_df_from_str(s)
    dfp.columns
    print(dfp)
    print("finished run")


def describe_nulls(df: pd.DataFrame,
                   print_nulls_df = True
                   ) -> Union[pd.DataFrame, pd.Series]:
    num_nulls = df.isnull().sum()
    nulls_only: pd.Series = num_nulls[num_nulls > 0]
    nulls_only.sort_values(inplace=True)
    if print_nulls_df:
        print(nulls_only)
    return nulls_only

def get_num_cat_columns(
        df: pd.DataFrame,
        exclude: Union[str, Iterable[str], None] = None,
        print_summary = True
) -> (List[str], List[str]):
    """ returns a tuple of lists: (numerical columns, categorical columns)

    RECOMMENDED USAGE
    ----------
    cols_num, cols_cat = bk_lib.get_num_cat_columns(df, col_pred)

    Parameters
    ----------
    df:              pd.DataFrame
    exclude:         An optional column name or list/tuple/set of column names to be exclude from the list of column names returned (often b/c is the dependent variable to predict)
    print_summary:   Print the number of numerical, categorical, and excluded columns

    Examples
    --------
    >>> # __c Can run these directly in IPython!
    >>> cols_num, cols_cat = get_num_cat_columns(df, col_pred)
    >>> cols_num, cols_cat = get_num_cat_columns(df)
    >>> cols_num, cols_cat = get_num_cat_columns(df, ["SalePrice", "LogSalePrice"])
    """
    exclude = ["KEYeTPv: "] if exclude is None else\
        [exclude] if isinstance(exclude, str) else\
        list(exclude)
    assert isinstance(exclude, list)

    cols_cat = set(df.columns[df.dtypes == np.object].values)
    cols_cat -= set(exclude)
    cols_cat = list(cols_cat)

    cols_num = set(df.columns[df.dtypes != np.object].values)
    cols_num -= set(exclude)
    cols_num = list(cols_num)
    cols_num
    if print_summary:
        excluded_columns = set(df.columns) & set(exclude)
        columns_not_excluded_bc_non_existent = set(exclude) - set(df.columns)
        print(f"\ndef {get_num_cat_columns.__name__}: returning:\n\t"
              f"{len(cols_num)} numerical columns,\n\t"
              f"{len(cols_cat)} categorical columns. \n\t"
              f"EXCLUDED COLUMNS: {excluded_columns}\n\t"
              f"COLUMNS NOT EXCLUDED B/C NON-EXISTENT: {columns_not_excluded_bc_non_existent}")
    return cols_num, cols_cat

# __Q:  Import, update, and make it use the new fcn P4t170910 P5t171104 P5t180417 P6t190822 P7t230910 P7t351109 rm3.0 tp5
def practice_re_importing():
    """
    USAGE
    ---------
    bk_lib.practice_re_importing()

    :return:
    """
    print("practice_re_importing ID is 2")

    if _ANS:
        import importlib
        importlib.reload(bk_lib)


class FeatherManager:
    """Manages saving/loading to/from feather format

    GENERALLY I"M USING PICKLE FORMAT NOW"""
    def __init__(self, filename_getter: tp.Callable) -> None:
        """
        :type filename_getter: fcn(prefix)  ->  str

        fm = FeatherManager(feather_get_filepath)
        """
        super().__init__()
        self.filename_getter = filename_getter

    def feather_save_to(self, df: pd.DataFrame, prefix: str):
        """

        :param df:
        :param prefix:
        """
        df.to_feather(self.filename_getter(prefix))

    def feather_read_from(self, prefix) -> pd.DataFrame:
        """

        :param prefix:
        :return:
        """
        return pd.read_feather(self.filename_getter(prefix))

    def feathers_save_to(self, dfs: tp.List[pd.DataFrame], prefixes: tp.List[str]):
        """

        :param dfs:
        :param prefixes:
        """
        for df, prefix in zip(dfs, prefixes):
            self.feather_save_to(df, prefix)

    def feathers_read_from(self, prefixes):
        """

        :param prefixes:
        :return:
        """
        return [self.feather_read_from(prefix) for prefix in prefixes]