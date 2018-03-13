import os
import os.path as osp
import re
import sys
import enum
import typing as tp
import itertools as it
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable, Sized, NamedTuple, Optional
from functools import partial

def to_num(s: str)  ->  Union[int, float]:
    try:
        return int(s)
    except ValueError:
        return float(s)
# __c The fcn name autocompletes from the clipboard, so copy it first!
def TEST_to_num() -> None:
    """Simple tests of function "to_num" to help with development in debug mode, as well as for finding bugs"""
    assert isinstance(to_num("2"), int)
    assert isinstance(to_num("2.0"), float)
    assert to_num("2.0") == 2.0
    assert to_num("2.5") == 2.5
    assert to_num("-2.5") == -2.5
    print(f"PC:KEYgCvB:   TEST_to_num done")
    exit(1)
# if __name__ == '__main__':
#     TEST_to_num()