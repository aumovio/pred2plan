import functools
import time 
import logging 
import secrets


logger = logging.getLogger(__name__)


def timeit(func):
    """Decorator to measure execution time of a function and log it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"[TIMEIT] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

