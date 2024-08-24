"""This is an incomplete implementation of the UBJSON specification. Expected is a readable file pointer to a UBJSON file.
Things that are not implemented:
- High precision numbers
- Optimized arrays with type & count
- nested arrays, so arrays in arrays are not supported, objects in arrays are supported
"""

import struct
from typing import Any

import numpy as np


def b(s):
    return isinstance(s, str) and s.encode("latin1") or s


NOOP = b("N")
NULL = b("Z")
FALSE = b("F")
TRUE = b("T")
INT8 = b("i")
UINT8 = b("U")
INT16 = b("I")
INT32 = b("l")
INT64 = b("L")
FLOAT32 = b("d")
FLOAT64 = b("D")
CHAR = b("C")
STRING = b("S")
HIDEF = b("H")
ARRAY_OPEN = b("[")
ARRAY_CLOSE = b("]")
OBJECT_OPEN = b("{")
OBJECT_CLOSE = b("}")

INTEGERS = [INT8, UINT8, INT16, INT32, INT64]
FLOATS = [FLOAT32, FLOAT64]

type_sizes = {
    INT8: 1,
    UINT8: 1,
    INT16: 2,
    INT32: 4,
    INT64: 8,
    FLOAT32: 4,
    FLOAT64: 8,
    # todo: maybe add high-precision number
    CHAR: 1,
}

struct_mapping = {
    INT8: "b",
    UINT8: "B",
    INT16: "h",
    INT32: "i",
    INT64: "q",
    FLOAT32: "f",
    FLOAT64: "d",
    CHAR: "s",
}

numpy_type_mapping = {
    INT8: np.int8,
    UINT8: np.uint8,
    INT16: np.int16,
    INT32: np.int32,
    INT64: np.int64,
    FLOAT32: np.float32,
    FLOAT64: np.float64,
}


objects = [OBJECT_OPEN, OBJECT_CLOSE]
arrays = [ARRAY_CLOSE, ARRAY_OPEN]

# Can decode optimized array
# [[][#][i][5] // An array of 5 elements.
#     [d][29.97]
#     [d][31.13]
#     [d][67.0]
#     [d][2.113]
#     [d][23.8889]
# // No end marker since a count was specified.

# AND


# [[][$][d][#][i][5] // An array of 5 float32 elements.
#     [29.97] // Value type is known, so type markers are omitted.
#     [31.13]
#     [67.0]
#     [2.113]
#     [23.8889]
# // No end marker since a count was specified.
def _decode_array_optimized(fp):
    tag = fp.read(1)
    # optimized array with count
    if tag == b"#":
        array_length_indicator = fp.read(1)
        array_length_indicator_type = struct_mapping[array_length_indicator]
        array_length = type_sizes[array_length_indicator]
        array_length = struct.unpack(f">{array_length_indicator_type}", fp.read(array_length))[0]

        array = []
        for _ in range(array_length):
            tag = fp.read(1)
            element = __decode_element(tag, fp)
            array.append(element)
        return array
    # optimized array with type & count
    elif tag == b"$":
        value_type_byte = fp.read(1)
        value_type_length = type_sizes[value_type_byte]
        tag = fp.read(1)
        array_type_byte = fp.read(1)
        array_type_length = type_sizes[array_type_byte]
        array_type_prefix = struct_mapping[array_type_byte]
        array_length_bytes = fp.read(array_type_length)
        array_length = struct.unpack(f">{array_type_prefix}", array_length_bytes)[0]
        if array_length == 0:
            return list()
        buffer = fp.read(array_length * value_type_length)
        return list(struct.unpack(">" + f"{struct_mapping[value_type_byte]}" * array_length, buffer))
    else:
        raise ValueError("Expected optimized array but got received bytes of unoptimized array.")


def _decode_object(tag, fp):
    result_dict: dict[str, Any] = dict()
    if tag == OBJECT_OPEN:
        key_type = None
        while key_type != OBJECT_CLOSE:
            if key_type is None:
                key_type = fp.read(1)
                # case for empty object
                if key_type == OBJECT_CLOSE:
                    return {}
            if key_type == OBJECT_OPEN:
                return _decode_object(key_type, fp)
            key, value = _decode_simple_key_value_pair(fp, key_type)
            if key == "}":
                return result_dict
            result_dict[key] = value
            key_type = fp.read(1)
    return result_dict


def __decode_element(tag, fp):
    if (element_type_length := type_sizes.get(tag)) is not None:
        value_bytes = fp.read(element_type_length)
        value_struct_prefix = struct_mapping[tag]
        return struct.unpack(f">{value_struct_prefix}", value_bytes)[0]
    elif tag == STRING:
        string_length_type = fp.read(1)
        length = __decode_element(string_length_type, fp)
        string_bytes = fp.read(length)
        return string_bytes.decode("utf-8")
    elif tag == OBJECT_OPEN:
        return _decode_object(tag, fp)
    else:
        raise ValueError(f"Expected type size for {tag} but got {element_type_length}")


# Can decode
# [i][3][lat][d][29.976]
def _decode_simple_key_value_pair(fp, key_type):
    if key_type in type_sizes:
        length_of_key = __decode_element(key_type, fp)
        key_to_decode = fp.read(length_of_key)
        key: str = key_to_decode.decode("utf-8")
        value_type_byte = fp.read(1)
        if value_type_byte in type_sizes:
            value = __decode_element(value_type_byte, fp)
            return key, value
        elif value_type_byte == STRING:
            # todo: check for string, high precision number, array, object here
            # value_type_byte = __decode_element(value_type_byte, fp)
            value = __decode_element(value_type_byte, fp)
            return key, str(value)
        elif value_type_byte == OBJECT_OPEN:
            value = _decode_object(value_type_byte, fp)
            return key, value
        elif value_type_byte == OBJECT_CLOSE:
            return key, {}
        elif value_type_byte == ARRAY_OPEN:
            value = _decode_array_optimized(fp)
            return key, value
        elif value_type_byte == ARRAY_CLOSE:
            return key, []
        elif value_type_byte == b"" and key == "}":
            return key, None
        else:
            raise ValueError(f"Unmatched value type for {value_type_byte}.")
    else:
        raise ValueError(f"Expected type size for {key_type} but could not find any.")


def decode_ubjson_buffer(fp):
    fp.read(1)
    complete_dict = dict()
    key_type = fp.read(1)
    while key_type != b"" and key_type != OBJECT_CLOSE:
        key, value = _decode_simple_key_value_pair(fp, key_type)
        complete_dict[key] = value
        key_type = fp.read(1)
    return complete_dict
