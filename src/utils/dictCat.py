"""
Function to concatenate all elements of dict together
"""


def dict_cat(dict1: dict, dict2: dict) -> dict:
    """
    Concatenate all elements of two dicts together
    :param dict1: First dict
    :param dict2: Second dict
    :return: Concatenated dict
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    common_keys = keys1.intersection(keys2)
    unique_keys1 = keys1 - keys2
    unique_keys2 = keys2 - keys1

    def cat(x, y):
        if isinstance(x, list) and isinstance(y, list):
            return x + y
        elif isinstance(x, list):
            return x + [y]
        elif isinstance(y, list):
            return [x] + y
        else:
            return [x, y]

    def to_list(x):
        return x if isinstance(x, list) else [x]

    return ({key: cat(dict1[key], dict2[key]) for key in common_keys}
            | {key: to_list(dict1[key]) for key in unique_keys1}
            | {key: to_list(dict2[key]) for key in unique_keys2})


