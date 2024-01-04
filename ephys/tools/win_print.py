import functools

def winprint(func):
    """
    Wrapper decorator for functions that print to the text area Clears the print
    area first, and puts a line of '*' when the function returns
    """

    @functools.wraps(func)
    def wrapper_print(self, *args, **kwargs):
        self.textclear()
        value = func(self, *args, **kwargs)
        # end_time = time.perf_counter()      # 2 run_time = end_time -
        # start_time    # 3
        self.textappend("*" * 80)
        # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_print


def winprint_continuous(func):
    """
    Wrapper decorator for functions that print to the text area DOES NOT clear
    the print area first,
    """

    @functools.wraps(func)
    def wrapper_print(self, *args, **kwargs):
        value = func(self, *args, **kwargs)
        # end_time = time.perf_counter()      # 2 run_time = end_time -
        # start_time    # 3 print(f"Finished {func.__name__!r} in {run_time:.4f}
        # secs")
        return value

    return wrapper_print