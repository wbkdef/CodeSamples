"""
import bk_typing

"""


import typing as tp
from typing import Union, List, Tuple, Dict, Sequence, Iterable, TypeVar, Any


TV = TypeVar('TV')


# __t New Number Subtypes, and functions to cast to them
NonNegInt = tp.NewType('NonNegInt', int)
NonNegFloat = tp.NewType('NonNegFloat', float)
Probability = tp.NewType('Probability', NonNegFloat)
def NNI(n: int)  ->  NonNegInt:
    assert n >= 0, f"n should be cast-able to a NonNegInt, but is: {n}"
    return tp.cast(NonNegInt, n)
def NNF(f: float)  ->  NonNegFloat:
    assert f >= 0, f"f should be cast-able to a NonNegFloat, but is: {f}"
    return tp.cast(NonNegFloat, f)
def PBT(p: float)  ->  Probability:
    assert p >= 0 and p <= 1, f"p should be cast-able to a Probability, but is: {p}"
    return tp.cast(Probability, p)
def TEST_New_Number_Subtypes():
    pass
    # __WARN: These don't actually check (with mypy) while inside this function - dedent then run mypy to check that all working!
        # They do check if inside the """if __name__ == '__main__':""", but not inside this function!
    if "IPYTHON" == "don't run":
        NNI(-1)  # __c Should throw assertion errors when run
        NNF(-.5)  # __c Should throw assertion errors when run
        PBT(1.5)  # __c Should throw assertion errors when run
    NNI(1)
    NNF(2.5)
    PBT(.35)

    def ThroughProb(p: Probability)  ->  Probability:
        return p

    ThroughProb(.5)  # __c Expect error

    x: NonNegFloat
    x = PBT(.6)
    y: Probability
    y = NonNegFloat(.6)  # __c Expect error
    y = x  # __c Expect error
    x = y
    x = .5  # __c Expect error
    y = 9.99  # __c Expect error


# __t New Iterable (or self) Types
ListTuple = Union[List[TV], Tuple[TV, ...]]
SelfList = Union[TV, List[TV]]
SelfTuple = Union[TV, Tuple[TV]]
SelfListTuple = Union[TV, ListTuple[TV]]
SelfSequence = Union[TV, Sequence[TV]]
SelfIterable = Union[TV, Iterable[TV]]


# __c Recursion/Forward Referencing of types not supported, so use "Any" at 2nd level
# __c Also, "List" is an invariant type, so 2nd level types don't work that well as parameter annotations (cause have to match exactly, not just subclass)
SelfList_Recursive     = Union[TV, List[        Union[TV, List[        Any]]]]
SelfSequence_Recursive = Union[TV, Sequence[    Union[TV, Sequence[    Any]]]]
SelfIterable_Recursive = Union[TV, Iterable[    Union[TV, Iterable[    Any]]]]



# if "Test Self____" == "Skip":
if True:
    # Typed fcns
    def my_fcn_SelfList(inp: SelfList[int]): pass
    def my_fcn_SelfListTuple(inp: SelfListTuple[int]): pass
    def my_fcn_SelfSequence(inp: SelfSequence[int]): pass
    def my_fcn_SelfIterable(inp: SelfIterable[int]): pass
    def my_fcn_SelfList_Recursive(inp: SelfList_Recursive[int]): pass
    # def my_fcn_SelfListTuple_Recursive(inp: SelfListTuple_Recursive[int]): pass
    def my_fcn_SelfSequence_Recursive(inp: SelfSequence_Recursive[int]): pass
    def my_fcn_SelfIterable_Recursive(inp: SelfIterable_Recursive[int]): pass

    an_int: int = 3
    list_int: List[int] = [3, 4]
    list_list_int: List[Union[int, List[int]]] = [3, 9, [4, 5]]
    tuple_int: Tuple[int, ...] = (3, 4)
    tuple_tuple_int: Tuple[Union[int, Tuple[int, ...]], ...] = (3, 9, (4, 5))
    list_tuple_int: List[Union[int, Tuple[int, ...]]] = [3, 9, (4, 5)]  # todo ADD this!
    iter_int: Iterable[int] = range(5)
    list_iter_int: List[Union[int, Iterable]] = [3, 9, range(5)]

    # ______t Test  List
    my_fcn_SelfList(an_int)
    my_fcn_SelfList(list_int)
    # my_fcn_SelfList(list_list_int) # __c Should complain
    # my_fcn_SelfList(tuple_int) # __c Should complain
    # my_fcn_SelfList(tuple_tuple_int) # __c Should complain
    # my_fcn_SelfList(list_tuple_int) # __c Should complain
    # my_fcn_SelfList(iter_int) # __c Should complain
    # my_fcn_SelfList(list_iter_int) # __c Should complain

    # ______t Test  ListTuple
    my_fcn_SelfListTuple(an_int)
    my_fcn_SelfListTuple(list_int)
    # my_fcn_SelfListTuple(list_list_int) # __c Should complain
    my_fcn_SelfListTuple(tuple_int)
    # my_fcn_SelfListTuple(tuple_tuple_int) # __c Should complain
    # my_fcn_SelfListTuple(list_tuple_int) # __c Should complain
    # my_fcn_SelfListTuple(iter_int) # __c Should complain
    # my_fcn_SelfListTuple(list_iter_int) # __c Should complain

    # ______t Test  Sequence
    my_fcn_SelfSequence(an_int)
    my_fcn_SelfSequence(list_int)
    # my_fcn_SelfSequence(list_list_int) # __c Should complain
    my_fcn_SelfSequence(tuple_int)
    # my_fcn_SelfSequence(tuple_tuple_int) # __c Should complain
    # my_fcn_SelfSequence(list_tuple_int) # __c Should complain
    # my_fcn_SelfSequence(iter_int) # __c Should complain
    # my_fcn_SelfSequence(list_iter_int) # __c Should complain

    # ______t Test  Iterable
    my_fcn_SelfIterable(an_int)
    my_fcn_SelfIterable(list_int)
    # my_fcn_SelfIterable(list_list_int) # __c Should complain
    my_fcn_SelfIterable(tuple_int)
    # my_fcn_SelfIterable(tuple_tuple_int) # __c Should complain
    # my_fcn_SelfIterable(list_tuple_int) # __c Should complain
    my_fcn_SelfIterable(iter_int)
    # my_fcn_SelfIterable(list_iter_int) # __c Should complain

    # __t RECURSIVE
    # # ______t Test  List_Recursive
    my_fcn_SelfList_Recursive(3)
    my_fcn_SelfList_Recursive([3, 4])
    my_fcn_SelfList_Recursive([3, 9, [4, 5]])  # __c Should be OK b/c passing in directly, so isn't concerned about it being mutated incorrectly
    # my_fcn_SelfList_Recursive((3, 4))  # __c Should complain
    # my_fcn_SelfList_Recursive((3, 9, (4, 5)))  # __c Should complain
    # my_fcn_SelfList_Recursive([3, 9, (4, 5)])  # __c Should complain
    # my_fcn_SelfList_Recursive(range(5))  # __c Should complain
    # my_fcn_SelfList_Recursive([3, 9, range(5)])  # __c Should complain

    # ______t Test  Sequence_Recursive
    my_fcn_SelfSequence_Recursive(an_int)
    my_fcn_SelfSequence_Recursive(list_int)
    my_fcn_SelfSequence_Recursive(list_list_int)
    my_fcn_SelfSequence_Recursive(tuple_int)
    my_fcn_SelfSequence_Recursive(tuple_tuple_int)
    my_fcn_SelfSequence_Recursive(list_tuple_int)
    # my_fcn_SelfSequence_Recursive(iter_int) # __c Should complain
    # my_fcn_SelfSequence_Recursive(list_iter_int) # __c Should complain

    # ______t Test  Iterable_Recursive
    my_fcn_SelfIterable_Recursive(an_int)
    my_fcn_SelfIterable_Recursive(list_int)
    my_fcn_SelfIterable_Recursive(list_list_int)
    my_fcn_SelfIterable_Recursive(tuple_int)
    my_fcn_SelfIterable_Recursive(tuple_tuple_int)
    my_fcn_SelfIterable_Recursive(list_tuple_int)
    my_fcn_SelfIterable_Recursive(iter_int)
    my_fcn_SelfIterable_Recursive(list_iter_int)





