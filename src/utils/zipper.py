"""
Enhanced zip function which deals with broadcasting
"""


def zipper(*args):
    try:
        length = max(len(item) for item in args if hasattr(item, '__iter__') and not isinstance(item, str))
    except ValueError:
        length = 1
    return zip(*((item if len(item) == length else length * item)
                 if hasattr(item, '__iter__') and not isinstance(item, str) else length * [item]
                 for item in args))
