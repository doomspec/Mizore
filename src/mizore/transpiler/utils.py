from typing import List, Dict

def get_content(obj_list: List, prefix=None):
    if prefix is None:
        return obj_list[-1]

    for obj in obj_list[::-1]:
        if obj.name.startswith(prefix):
            return obj


def get_content_by_type(obj_list: List, type_name):
    for obj in obj_list[::-1]:
        if type_name in obj.types:
            return obj
    raise Exception(f"Object with type {type_name} is not found")


def append_dict_items(log: Dict, output: Dict):
    for key, value in log.items():
        if key == "name" or key == "types":
            continue
        output[key].append(value)
