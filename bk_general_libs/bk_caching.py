"""
from bk_general_libs import bk_caching

# __Q:  Run pytest on this P2t180313 P3t180430 P4t181107 P5t201209 P6t290419 rm4.0 tp5
# __A:  # Navigate to the test file
# __A:  Run (q & 4)
# todo_weekly:  Lint-C
"""

import pickle

import os
import os.path as osp
import re
import sys
import typing as tp
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable, Sized

# import sklearn as sk

import bk_general_libs.bk_typing as tp_bk #tbk
from bk_general_libs import bk_strings
from bk_general_libs import bk_context_managers
import functools

from bk_general_libs.test_bk_caching import test_PickleCached

T = tp.TypeVar('T')


def filename_str_from_dict(d: tp.Dict)  ->  str:
    """Turns dict "d" into a string (to be used as part of a filename)"""
    return "_" + "_".join(f"{k}{d[k]}" for k in sorted(d.keys()))

# todo switch to supycache with redis?  https://pypi.python.org/pypi/supycache
# (memcache can store max 250 bytes, redis 512 MB, so redis more useful)
# redis appears to assume you're on Linux - doesn't look Windows-friendly!  (Back end of webpages)
# todo Implement some supycache features?
# 1)  Passing a format string or callable to create the hash
# 2)  A time delay, after which it is recalled (a day for me?)

class PickleCached:
    def __init__(self, *, description: str = "", pickles_dir: str = None, print_updates=True) -> None:
        r""" A decorator that implements pickle file-backed caching of a function

        A decorated function will load from a generated pickle file if possible,
        else run the function and save the result to that same pickle file.

        :param description:
        :param pickles_dir:

        Example:
            @PickleCached(description="GSDesc5")  # __c To ignore previous caches, change the description or delete the pickles files
            def get_str(name, *, age):
                import time
                time.sleep(.0003456)
                print(f"PC:KEYogcp:  in get_str  ->  name, age is [[{name, age}]]")
                return f"I'm {name}, aged {age}"

            get_str("Bruce", age=36)
              ->
                PC:KEYdQte: Failed to PICKLE LOAD from C:\Users\wbruc\Desktop\git_repos\organization\bk_python_libs\bk_general_libs\pickles\GSDesc5_get_str_Bruce___age36.pickle, e: [Errno 2] No such file or directory: 'C:\\Users\\wbruc\\Desktop\\git_repos\\organization\\bk_python_libs\\bk_general_libs\\pickles\\GSDesc5_get_str_Bruce___age36.pickle'.  Will be calling decorated function
                PC:KEYogcp:  in get_str  ->  name, age is [[('Bruce', 36)]]
                KEYuXCo:  Took [[0.002]] seconds to 'Call decorated function & pickle results to C:\Users\wbruc\Desktop\git_repos\organization\bk_python_libs\bk_general_libs\pickles\GSDesc5_get_str_Bruce___age36.pickle'
                Out[10]: "I'm Bruce, aged 36"

            get_str("Bruce", age=36)  # __c The second time it just uses the cached value
              ->
                KEYuXCo:  Took [[0.001]] seconds to 'load cache function result via decorator from C:\Users\wbruc\Desktop\git_repos\organization\bk_python_libs\bk_general_libs\pickles\GSDesc5_get_str_Bruce___age36.pickle'
                Out[11]: "I'm Bruce, aged 36"
        """
        self.description = description
        self.pickles_dir = pickles_dir if pickles_dir is not None else osp.join(os.getcwd(), "pickles")
        self.print_updates = print_updates
        super().__init__()

    def __call__(self, fcn_to_decorate: Callable)  ->  Callable:
        @functools.wraps(fcn_to_decorate)
        def pickle_cached_fcn(*args, **kwargs):
            # get pickle file's path
            os.makedirs(self.pickles_dir, exist_ok=True)
            args_str = bk_strings.get_1_line_iterable_representation(args, items_at_start_and_end=30, items_sep="_")
            dicts_str = filename_str_from_dict(kwargs)
            pickle_file_path = osp.join(self.pickles_dir, f"{self.description}_{fcn_to_decorate.__name__}_{args_str}__{dicts_str}.pickle")

            # Try to load from pickle
            try:
                with bk_context_managers.print_time(f"load cache function result via decorator from {pickle_file_path}", print_at_start=False, print_at_end=self.print_updates):
                    with open(pickle_file_path, "rb") as f:
                        return pickle.load(f)
            except FileNotFoundError as e:
                if self.print_updates:
                    print(f"PC:KEYdQte: Failed to PICKLE LOAD from {pickle_file_path}, e: {e}.  Will be calling decorated function")

            # Calling decorated function itself
            with bk_context_managers.print_time(f"Call decorated function & pickle results to {pickle_file_path}", print_at_start=self.print_updates, print_at_end=self.print_updates):
                obj = fcn_to_decorate(*args, **kwargs)
                # Save to pickle
                with open(pickle_file_path, "wb") as f:
                    pickle.dump(obj, f)

            return obj
        return pickle_cached_fcn
if __name__ == '__main__':
    test_PickleCached()


class PickleManager:
    """Manages saving/loading to/from pickle format

    GENERALLY I"M USING PICKLE FORMAT NOW"""
    def __init__(self, pickles_dir: str, run_desc: str, load_enabled=True, **kwargs) -> None:
        """run_desc and **kwargs will be used to create file name of all files saved/loaded

        :param pickles_dir:  The directory to save/load pickles
        :param run_desc:  A description to include as part of all file names
        :param load_enabled:  Whether loading from pickle is enabled (usually disabled to calculate everything from scratch)
        :param kwargs: A dict that will also be turned into "A description to include as part of all file names"
        """
        super().__init__()
        self.load_enabled = load_enabled
        self.pickles_dir = pickles_dir
        self.run_desc = run_desc + filename_str_from_dict(kwargs)

        os.makedirs(pickles_dir, exist_ok=True)

    def get_filename(self, suffix: str, **kwargs)  ->  str:
        """Returns the filename to pickle save/load to/from"""
        return osp.join(self.pickles_dir, self.run_desc + "__" + suffix + filename_str_from_dict(kwargs) + ".pickle")

    def load(self, suffix, **kwargs)  ->  tp.Optional[tp.Any]:
        """Loads the Pickled object.

        The suffix and **kwargs become part of the filename that will be loaded"""
        file_name = self.get_filename(suffix, **kwargs)
        if not self.load_enabled:
            print(f"PC:KEYcxKs: PICKLE LOAD DISABLED (from {file_name})")
            return None
        try:
            with open(file_name, "rb") as f:
                print(f"PC:KEYiLhH:  PICKLE LOADed from {file_name}")
                val = pickle.load(f)
                return val
        # pylint: disable=broad-except
        except FileNotFoundError as e:
            print(f"PC:KEYiLhH:  Failed to PICKLE LOAD from {file_name}, e: {e}")
            return None

    def save(self, obj, suffix: str, **kwargs)  ->  None:
        """Pickles the object.

        The suffix and **kwargs become part of the filename"""
        file_name = self.get_filename(suffix, **kwargs)
        with open(file_name, "wb") as f:
            pickle.dump(obj, f)
            print(f"PC:KEYmFOk:  PICKLE SAVED to {file_name}")

    def get_memoized_result_no_args(self, *, fcn: tp.Callable[[], T], suffix: str)  ->  T:
        """Calls fcn with **kwargs, memoized to pickle filename based on 'suffix'

         - This is recommended for more complicated calls
         - Requires building the suffix manually

        Example:
            data_nietszche: bk_data_sets.TextData = setup.PM.get_memoized_result_no_args(
                fcn=lambda: bk_data_sets.TextData(bk_data_sets.DataSetNietzsche.get_raw(), x_bigram_size=setup.n_x_chars, shorten_text_to_len=setup.data_subset_size),
                suffix=suffix
            )


            data_nietszche = setup.PM.get_memoized_result(bk_data_sets.NietszcheData, "NietszcheData", n_x_chars=setup.n_x_chars, subset_size=setup.data_subset_size)
        If needed, under the hood this calls:
            bk_data_sets.NietszcheData(n_x_chars=setup.n_x_chars, subset_size=setup.data_subset_size)
        """
        res = self.load(suffix)
        if res is None:
            file_name = self.get_filename(suffix)
            print(f"in get_memoized_result, unable to load, so calling {fcn.__name__} from scratch: {file_name}")
            res = fcn()
            self.save(res, suffix)
        return res

    def get_memoized_result(self, *, fcn: tp.Callable, suffix: str, **kwargs)  ->  tp.Any:
        """Calls fcn with **kwargs, memoized to pickle filename based on 'suffix' and '**kwargs'

        Example:
            data_nietszche = setup.PM.get_memoized_result(bk_data_sets.NietszcheData, "NietszcheData", n_x_chars=setup.n_x_chars, subset_size=setup.data_subset_size)
        If needed, under the hood this calls:
            bk_data_sets.NietszcheData(n_x_chars=setup.n_x_chars, subset_size=setup.data_subset_size)
        """
        suffix = suffix + filename_str_from_dict(kwargs)
        fcn = lambda: fcn(**kwargs)

        res = self.get_memoized_result_no_args(fcn=fcn,  suffix=suffix)
        return res

    @staticmethod
    def test()  ->  None:
        """Tests this Class"""
        PM = PickleManager(r"C:\Users\wbruc\Desktop\git_repos\organization\bk_python_libs\bk_ds_libs\test_data", "cool_fn", one=2, Three="Four")
        file_name = PM.get_filename('MySuffix', five='alpha')
        print(f"PC:KEYvunu:  {file_name}")
        assert file_name == r"C:\Users\wbruc\Desktop\git_repos\organization\bk_python_libs\bk_ds_libs\test_data\cool_fn_ThreeFour_one2__MySuffix_fivealpha.pickle"

        assert PM.load("dne") is None

        PM.save([1, 2, "three", "four"], "A_file", xk=8)
        lst = PM.load("A_file", xk=8)
        print(f"PC:KEYuGAk:  lst{lst}")
        assert lst == [1, 2, "three", "four"]
        # assert do == {1:2, "three":"four"}

