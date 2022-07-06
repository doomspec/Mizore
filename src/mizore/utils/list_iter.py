from typing import List, Iterable


def _flatten_list(lst, res: List):
    if isinstance(lst, List):
        if isinstance(lst[0], List):
            sub_len = len(lst[0])
            for sub_lst in lst:
                assert len(sub_lst) == sub_len
                _flatten_list(sub_lst, res)
        else:
            res.extend(lst)
    else:
        res.append(lst)


def list_shape(lst: List):
    shape = []
    sub_lst = lst
    while isinstance(sub_lst, List):
        shape.append(len(sub_lst))
        sub_lst = sub_lst[0]
    return shape


def flatten_list(lst):
    if not isinstance(lst, List):
        return lst, []
    res = []
    _flatten_list(lst, res)
    return res, list_shape(lst)


def _reshape_flattened(start, end, flattened, shape_start, shape):
    if shape_start == len(shape) - 1:
        return flattened[start: end]
    this_shape = shape[shape_start]
    res = []
    sub_length = (end - start) // this_shape
    for i in range(this_shape):
        res.append(
            _reshape_flattened(start + i * sub_length, start + (i + 1) * sub_length, flattened, shape_start + 1, shape))
    return res


def reshape_flattened(flattened, shape):
    if len(shape) == 0:
        return flattened
    else:
        return _reshape_flattened(0, len(flattened), flattened, 0, shape)


if __name__ == '__main__':
    print(reshape_flattened([[1, 2], [1, 2], [1, 2], [1, 2]], (2, 2)))
