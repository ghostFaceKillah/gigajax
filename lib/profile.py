import contextlib
import time


@contextlib.contextmanager
def easy_profile():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"It took {end-start} to run it")


if __name__ == '__main__':
    with easy_profile():
        time.sleep(1.)
