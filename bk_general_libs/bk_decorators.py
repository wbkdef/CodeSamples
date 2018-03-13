import collections
import functools


# ______t Get signature hash
# https://stackoverflow.com/questions/10220599/how-to-hash-args-kwargs-for-function-cache
kwd_mark = object()     # sentinel for separating args from kwargs

def create_signature_hashable(*args, **kwargs):
    # __c Note - if some of the arguments are lists, this still won't be hashable
    key = args + (kwd_mark,) + tuple(sorted(kwargs.items()))
    assert isinstance(key, collections.Hashable) # __c This is not reliable
    key.__hash__()  # Double check hashable
    d = {key: "triple check that hashable, and use-able as dictionary key"}
    return key
def TEST_create_signature_hashable():
    key1 = create_signature_hashable(5, "cool")
    key2 = create_signature_hashable(5, "cool", kwa="my arg")
    key3 = create_signature_hashable(5, "cool", kwa="my arg", lst=[2, 5])
    key4 = create_signature_hashable(5, "cool", [2, 5], kwa="my arg")
    key1, key2, key3
    {key1:1}
    {key1:1, key2: 2}
    {key1:1, key2: 2, key3: 3}
# if __name__ == '__main__':
#     TEST_create_signature_hashable()


# ______t A memoization decorator (in memory - requires hashable arguments)
# # __c Consider using rlu_cache instead!
def memoized(func):
    cache = {}

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        key = create_signature_hashable(*args, **kwargs)
        # if not isinstance(args, collections.Hashable):
        #     print(f"PC:KEYolGP:  args not hashable.  can't memoize for args: {args}")
        #     return func(*args, **kwargs)
        # if not isinstance(kwargs, collections.Hashable):
        #     print(f"PC:KEYolGP:  kwargs not hashable.  can't memoize for kwargs: {kwargs}")
        #     return func(*args, **kwargs)

        if key in cache:
            print(f"PC:KEYwmzY:  Using cached value for args: {args}, kwargs: {kwargs}")
            return cache[key]
        print(f"PC:KEYwmzY:  Cached value DNE, calculating - for args: {args}, kwargs: {kwargs}")
        res = func(*args, **kwargs)
        cache[key] = res
        return res

    return wrapped_func
def TEST_memoized():
    @memoized
    def fibonacci(n):
        "Return the nth fibonacci number."
        print(f"PC:KEYqdyB:  n is {n}")
        if n in (0, 1):
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    print(f"\nfibonacci(6) is [[{fibonacci(6)}]]")
    print(f"\nfibonacci(9) is [[{fibonacci(9)}]]")
    fibonacci.__name__
    print(f"PC:KEYcBCZ:  Done TEST_memoized")
# if __name__ == '__main__':
#     TEST_memoized()


import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func
def TEST_deprecated():
    # Examples

    @deprecated
    def some_old_function(x, y):
        return x + y

    class SomeClass:
        @deprecated
        def some_old_method(self, x, y):
            return x + y

    ret = some_old_function(2, 3)
    ret
# if __name__ == '__main__':
#     TEST_deprecated()


def trace(func):
    """ Prints arguments and return values, each time fcn is run! """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print('%s(%r, %r) -> %r' %
              (func.__name__, args, kwargs, result))
        return result
    return wrapper

