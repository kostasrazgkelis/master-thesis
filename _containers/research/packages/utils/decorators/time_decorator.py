# timing_utils.py
import time
import functools


def timeit(method):
    """
    Decorator to measure execution time of class methods.
    Sets the elapsed time as a property on the instance (method.__name__ + '_time')
    and prints the time taken.
    """

    @functools.wraps(method)
    def timed(self, *args, **kwargs):
        start_time = time.time()
        result = method(self, *args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        setattr(self, f"{method.__name__}_time", elapsed)
        print(f"{method.__name__} took {elapsed:.4f} seconds")
        return result

    return timed
