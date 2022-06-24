from numbers import Number
from jaxlib.xla_extension import DeviceArray
from numpy import ndarray

def is_number(obj):
    return isinstance(obj, Number) or isinstance(obj, DeviceArray) or isinstance(obj, ndarray)