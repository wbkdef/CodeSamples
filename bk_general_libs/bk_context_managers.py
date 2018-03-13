import os
import os.path as osp
import re
import sys
import enum
import typing as tp
import itertools as it
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable, Sized, NamedTuple, Optional
from functools import partial
import time
import contextlib



@contextlib.contextmanager
def print_time(description: str, *, print_at_start=True, print_at_end=True):
    if print_at_start: 
        print(f"KEYqFfs:  Starting {description} ...")
    st = time.time()
    yield
    et = time.time()
    if print_at_end: 
        print(f"KEYuXCo:  Took [[{et-st:.2g}]] seconds to '{description}'")
# with print_time("a test"):
#     time.sleep(.6)
