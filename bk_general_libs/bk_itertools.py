""" Additional iterator tools

# done Pylint-C  2018-02-11
"""

import os
import os.path as osp
import re
import sys
import enum
import typing as tp
import itertools as it
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any, Callable, Sized, NamedTuple, Optional
from functools import partial
import collections


T = tp.TypeVar('T')


# __t Working with ngrams
def ngrams_unsafe(inp: Iterable[T], *, n: int)  ->  Iterable[tp.Deque[T]]:
    """Returns n-grams tuples (of length n)

    These n-grams are mutable, and so this generator could be messed up if the deque is mutated between yields
    """
    assert n > 0
    iterable = iter(inp)
    ngram = collections.deque(it.islice(iterable, n-1))
    for x in iterable:
        ngram.append(x)
        # print(f"\n''.join(ngram) is [[{''.join(ngram)}]]")
        yield ngram
        # yield tuple(ngram)
        ngram.popleft()
def ngrams(inp: Iterable[T], *, n: int)  ->  Iterable[Tuple[T, ...]]:
    """Returns n-grams tuples (of length n)"""
    for ngram in ngrams_unsafe(inp, n=n):
        yield tuple(ngram)
def str_ngrams(inp: str, *, n: int) -> List[str]:
    """ Returns n-grams from an input string

    :param inp:
    :param n:
    :return:

    Example:
        str_ngrams('hello there', n=3)
              ->  ['hel', 'ell', 'llo', 'lo ', 'o t', ' th', 'the', 'her', 'ere']
    """
    return [''.join(ngram) for ngram in ngrams(inp, n=n)]
def TEST_ngrams():
    """TEST"""
    res = [''.join(ngram) for ngram in ngrams('hello there', n=3)]
    res = str_ngrams('hello there', n=3)
    res
    assert res == ['hel', 'ell', 'llo', 'lo ', 'o t', ' th', 'the', 'her', 'ere']
if __name__ == '__main__':
    TEST_ngrams()

    # todo create test function
    # todo add assertion


# __t Slicing off the start and end of an iterator
def _tail(inp: Iterable[T], num_from_each_tail=int)  ->  Tuple[List[T], int]:
    """Returns "num_from_each_tail" items from the end of "inp", and the number of items omitted before that
    """
    assert num_from_each_tail >= 0
    iterable = iter(inp)
    res = collections.deque(it.islice(iterable, 0, num_from_each_tail))
    res
    num_omitted = 0
    for x in iterable:
        res.append(x)
        res.popleft()
        num_omitted += 1
    return list(res), num_omitted
def tail(inp: Iterable[T], num_from_each_tail=int)  ->  List[T]:
    """Returns "num_from_each_tail" items from the end of "inp"."""
    res, _ = _tail(inp, num_from_each_tail)
    return res
def TEST_tail():
    """TEST"""
    res = tail(iter(range(11)), 3)
    res
    assert res == [8, 9, 10]

    res = tail(iter(range(3)), 8)
    assert res == [0, 1, 2]
    res
    pass
def _tails(iterable: Iterable[T], *, num_from_each_tail=Union[int, Tuple[int, int]])  ->  Tuple[List[T], List[T], int]:
    """Get lists of the first few and last few items of an iterable, and the number omitted from the middle.

    If the iterable is short enough that none are omitted and the first few and last few items would overlap, shortens the second list so that items returned don't overlap

    :param iterable:  Iterable to return the first and last few from as lists
    :param num_from_each_tail:  Num to return from each tail
    :return:  (Items from start, Items from end, num items omitted from middle)
    """
    num_start, num_end = (num_from_each_tail, num_from_each_tail) if isinstance(num_from_each_tail, int) else num_from_each_tail
    iterator = iter(iterable)
    start = list(it.islice(iterator, 0, num_start))  # Convert to list before next line iterates more
    end, num_omitted = _tail(iterator, num_end)
    return start, end, num_omitted
def tails(iterable: Iterable, num_from_each_tail=Union[int, Tuple[int, int]], *, add_middle_elt_if_elts_omitted=False, middle_elt="...{num_omitted}...", element_wise_fcn: tp.Callable=lambda x: x)  ->  List:
    """Get an iterator over the first few and last few items of an iterables

    If the first few and last few would overlap, returns all elements"""
    start, end, num_omitted = _tails(iterable, num_from_each_tail=num_from_each_tail)

    start = [element_wise_fcn(elt) for elt in start]
    end = [element_wise_fcn(elt) for elt in end]

    if add_middle_elt_if_elts_omitted is False or num_omitted == 0:
        middle: List[str] = []
    else:
        middle = [middle_elt.format(**locals())]

    return start + middle + end
def TEST_tails():
    """TEST"""
    t = tails(range(10), 3)
    assert list(t) == [0, 1, 2, 7, 8, 9]
    t
    t = tails(range(10), (3, 4), add_middle_elt_if_elts_omitted=True, middle_elt="...")
    assert t == [0, 1, 2, "...", 6, 7, 8, 9]
    t
    t = tails(range(10), (3, 4), add_middle_elt_if_elts_omitted=True)
    assert t == [0, 1, 2, "...3...", 6, 7, 8, 9]
    t
    t = tails(range(10), (3, 4), add_middle_elt_if_elts_omitted=True, element_wise_fcn=lambda x: 11 * x)
    assert t == [0, 11, 22, "...3...", 66, 77, 88, 99]
    t
    t = tails(range(5), (3, 4), add_middle_elt_if_elts_omitted=True)
    assert t == [0, 1, 2, 3, 4]
    t
if __name__ == '__main__':
    TEST_tail()
    TEST_tails()


# __t Functions copied from elsewhere
def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in it.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def split_by_idxs(seq: Sequence[T], idxs: Iterable[int])  ->  tp.Iterator[Sequence[T]]:
    """ Splits a sequence at the given idxs into a bunch of sequences """
    last, sl = 0, len(seq)
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]



# __t Mathematical Sequences
def uniform_deterministic()  ->  tp.Iterator[float]:
    """ Deterministically returns uniformely distributed values

    i.e. list(it.islice(uniform_deterministic(), 0, 6))
      ->  [.5, .25, .75, .125, .625, .375]
    """
    vals: List[float] = [0]
    offset = .5
    while True:
        new_vals = [val + offset for val in vals]
        vals.extend(new_vals)
        offset /= 2
        for val in new_vals:
            yield val
def TEST_uniform_deterministic() -> None:
    """Simple tests of function "uniform_deterministic" to help with development in debug mode, as well as for finding bugs"""
    res = list(it.islice(uniform_deterministic(), 0, 6))
    assert res == [.5, .25, .75, .125, .625, .375]
    print(f"PC:KEYggLG:   TEST_uniform_deterministic done")
    exit(1)
# if __name__ == '__main__':
#     TEST_uniform_deterministic()

def power_law_val(cdf_val: float, *, p: float, min_x: float, )  ->  float:
    """

    Given the probability distribution:
        f_x = c x**(-p)    for x > min_x
    then, the integral is:
        F_x = c/(-p+1) x**(-p+1)    for x > min_x
    so
        -c/(-p+1) min_x**(-p+1) == 1
        c/(-p+1) == -1 / min_x**(-p+1)
        c == (p-1) / min_x**(-p+1)
        F_x = 1 - [1 / min_x**(-p+1)] * x**(-p+1)
    given a uniform R.V. U, we can then get a power_law RV. by:
        U = 1 - [1 / min_x**(-p+1)] * X**(-p+1)
        X = [(1-U) * min_x**(-p+1)] ** 1/(-p+1)
        X = (1-U)**[1/(-p+1)] * min_x

    :param p:
    :param min_x:
    :param cdf_val: The U in the above equation
    :return:
    """
    return (1-cdf_val)**(-1/(p-1)) * min_x

def power_law_distributed_vals(*, p: float, min_x: float)  ->  Iterable[float]:
    """ Deterministically returns x-values distributed as:  c x**(-p)  for x > min_x """
    return (power_law_val(U, p=p, min_x=min_x) for U in uniform_deterministic())
    # U = uniform_deterministic()
    # while True:
    #     yield power_law_val(p=p, min_x=min_x, cdf_val=next(U))
def TEST_power_law_distributed_vals() -> None:
    """Simple tests of function "power_law_distributed_vals" to help with development in debug mode, as well as for finding bugs"""
    # vals = list(it.islice(power_law_distributed_vals(2, 3), 0, 100))
    import numpy as np
    num_vals = 1000
    vals = np.array(list(it.islice(power_law_distributed_vals(p=2, min_x=3), 0, num_vals)))
    vals
    hist = np.histogram(vals, [1, 3, 6, 12, 24, 48, 96, 1000000])
    assert (hist[0][:4] == [0, num_vals/2, num_vals/4, num_vals/8]).all()
    print(f"PC:KEYuFfH:   TEST_power_law_distributed_vals done")
    exit(1)
# if __name__ == '__main__':
#     TEST_power_law_distributed_vals()


def all_equal(iterator: Iterable, *,
              key: Callable[[Any], Any] = lambda x: x):
    iterator = iter(iterator)
    try:
        first = key(next(iterator))
    except StopIteration:
        return True  # If the iterable is empty, they are definitely all equal!
    return all(first == key(rest) for rest in iterator)
# assert all_equal([1, 1, 1]) == True
# assert all_equal([1, 1, 1, 2]) == False
# assert all_equal([[1, 1, 2], [1, 1, 1]]) == False
# assert all_equal([[1, 1, 2], [1, 1, 1]], key=len) == True
# assert all_equal([[1, 1, 2], [1, 1, 1]], key=sum) == False
# assert all_equal([[1, 1, 2], [2, 1, 1]], key=sum) == True

def all_equal_len(iterator: Iterable):
    return all_equal(iterator, key=len)

