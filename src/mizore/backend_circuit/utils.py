from typing import Tuple


def merge_qset(qset1: Tuple[int,...], qset2: Tuple[int,...]):
    """

    """
    merged_set = set(qset1).union(qset2)
    return tuple(merged_set)


if __name__ == "__main__":
    print(merge_qset((1,2,3),(3,2,1,4)))