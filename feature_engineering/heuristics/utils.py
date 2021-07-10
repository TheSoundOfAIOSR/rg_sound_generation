from time import time


def how_long(f):
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f"{f.__name__} execution takes {end - start: .4f} ms")
        return result
    return wrapper
