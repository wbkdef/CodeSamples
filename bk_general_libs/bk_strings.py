""" MODIFY DESCRIPTION

# done Pylint-C 2018-02-11
"""
# __t Standard Header

import os
import os.path as osp
import re
import sys
import typing as tp
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, NamedTuple, Callable

import bk_general_libs.bk_typing as tpbk
from bk_general_libs import bk_itertools

import string

# __t Header Done


def join_apply(iterable: Iterable, *, join_with=', ', apply: Callable[..., str] = str):
    return join_with.join(apply(v) for v in iterable)
# res = join_apply([1, 2, 6, 3])
# res

def get_1_line_iterable_representation(iterable: Iterable, *, items_at_start_and_end=5, item_stringifier: tp.Callable[[Any], str]=str, items_sep=",  ")  ->  str:
    """Get a 1-line string summary of a dict

      :param iterable:                 Iterable to summarize as a string
      :param items_at_start_and_end:   Num items to show from the start and end of the iterable.  The items in-between are not displayed.
      :param item_stringifier:         Function used to convert each item to a string
      :param items_sep:                String separating each item in the string representation of the iterable
      :return:

    EXAMPLES:
    get_1_line_iterable_representation(range(10), items_at_start_and_end=2, item_stringifier=str, items_sep="_")
      ->  "0_1_ ...6... _8_9"
    """
    tails = bk_itertools.tails(iterable, items_at_start_and_end, add_middle_elt_if_elts_omitted=True, element_wise_fcn=item_stringifier)
    res = items_sep.join(tails)
    return res
def TEST_get_1_line_iterable_representation():
    """TEST"""
    res = get_1_line_iterable_representation(range(10), items_at_start_and_end=2, item_stringifier=str, items_sep="_")
    res
    assert res == "0_1_ ...6... _8_9"

    res = get_1_line_iterable_representation(range(9), items_at_start_and_end=3, item_stringifier=lambda x: str(x*11), items_sep=", ")
    res
    assert res == "0, 11, 22,  ...3... , 66, 77, 88"

    res = get_1_line_iterable_representation(range(9), items_at_start_and_end=3, item_stringifier=lambda x: f"{x}"*2, items_sep=", ")
    res
    assert res == "00, 11, 22,  ...3... , 66, 77, 88"

    pass
# if __name__ == '__main__':
#     TEST_get_1_line_iterable_representation()


def get_1_line_dict_representation(d: tp.Dict, *, items_at_start_and_end=3, key_val_sep=": ", items_sep=",  ")  ->  str:
    """Get a 1-line string summary of a dict

    :param d:
    :param items_at_start_and_end:
    :param key_val_sep:
    :param items_sep:
    :return:

    EXAMPLES:
    get_1_line_dict_representation(dict(enumerate('abcdefghijklmnopqrstuvwxyz', -13)))
      ->  '-13: a,  -12: b,  -11: c,  10: x,  11: y,  12: z'
    """
    lst = [f"{key}{key_val_sep}{val}" for key, val in d.items()]
    res = get_1_line_iterable_representation(lst, items_at_start_and_end=items_at_start_and_end, items_sep=items_sep)
    return res
    # items = []
    # for i, (key, val) in enumerate(d.items()):
    #     # print(f"i, (key, val) is [[{i, (key, val)}]]")
    #     if i in range(items_at_start_and_end) or i in range(len(d))[-items_at_start_and_end:]:
    #         items.append(f"{key}{key_val_sep}{val}")
    # ret = items_sep.join(items)
    # ret
    # return ret
def TEST_get_1_line_dict_representation():
    """TEST"""
    # get_1_line_dict_representation(dict(enumerate('abcdefghijklmnopqrstuvwxyz', -13)))
    res = get_1_line_dict_representation(dict(enumerate('abcdefghi', -2)), items_at_start_and_end=3, key_val_sep=": ", items_sep=",  ")
    assert res == '-2: a,  -1: b,  0: c,  4: g,  5: h,  6: i'
    assert res == '-2: a,  -1: b,  0: c,   ...3... ,  4: g,  5: h,  6: i'
    res
# if __name__ == '__main__':
#     TEST_get_1_line_dict_representation()


def get_random_key(include_colon_and_spaces=False)  ->  str:
    import random
    char_options = string.ascii_letters + string.digits
    key = "KEY" + random.choice(string.ascii_lowercase) + ''.join(random.choice(char_options) for i in range(3))
    if include_colon_and_spaces:
        key = " %s: " % key

    return key
# print(f"\n[get_random_key() for i in range(10)] is [[{[get_random_key() for i in range(10)]}]]")
# [get_random_key() for i in range(10)]

def format_file_name(s: str)  ->  str:
    """Take a string and return a valid filename constructed from the string.

    Adapted from https://gist.github.com/seanh/93666

    Uses a whitelist approach: any characters not present in valid_chars are
    removed. valid_chars are all the letters and numbers, plus "_."

    Also spaces/dashes are replaced with underscores.

    Note: this method may produce invalid filenames such as ``, `.` or `..`
    When I use this method I prepend a date string like '2009_01_15_19_46_32_'
    and append a file extension like '.txt', so I avoid the potential of using
    an invalid filename.
    """
    file_name = s
    del s
    # print(f"\nfile_name is [[{file_name}]]")

    file_name = file_name.replace("-", "_")
    file_name = file_name.replace(" ", "_")
    # print(f"\nfile_name is [[{file_name}]]")

    valid_chars = f"_.{string.ascii_letters}{string.digits}"
    file_name = ''.join(c for c in file_name if c in valid_chars)
    # print(f"\nfile_name is [[{file_name}]]")

    return file_name
# res = format_filename("The spaces and-dashes should be replaced by underscores")
# print(f"\nres is [[{res}]]")
# assert format_filename("The spaces and-dashes should be replaced by underscores") == "The_spaces_and_dashes_should_be_replaced_by_underscores"
# assert format_filename("The &909234U( spaces HERE (should be)replaced by underscores") == 'The_909234U_spaces_HERE_should_bereplaced_by_underscores'
