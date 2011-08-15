from time import time
import sys
from contextlib import contextmanager

def split_list(original_list, n):
    """Split a list in N equal-sized bits.

    :param original_list: the list to split
    :param n: the number of wanted bits
    """
    if len(original_list) <= n:
        final_list = [original_list, ]
    else:
        final_list = []
        bit_size = len(original_list) / n
        for i in range(n):
            final_list.append(original_list[i*bit_size:(i+1)*bit_size])

    return final_list


@contextmanager
def mesure(what, print_f=None, indent=0, **kwargs):
    """Mesure the execution time of an operation. It should be used like this::

        with mesure("Description of the operation"):
            print "yeah"

    :param what: the description of what is being mesured
    :param print_f: a callable that will be called to display the messages
    :param **kwargs: additional parameters to give to the callable
    """
    def _print(text):
        print "  " * indent, text

    if not print_f:
        print_f = _print
    t0 = time()
    yield
    print_f("%s performed in %s" % (what, time() - t0), **kwargs)
