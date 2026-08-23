from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Any

MAX_STRING = 16 * 1024 * 1024
MAX_ARRAY = 10_000_000


def _read(fmt: str, handle: BinaryIO) -> Any:
    size = struct.calcsize(fmt)
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("unexpected end of GGUF metadata")
    return struct.unpack("<" + fmt, value)[0]


def _string(handle: BinaryIO) -> str:
    length = _read("Q", handle)
    if length > MAX_STRING:
        raise ValueError("GGUF string is too large")
    return handle.read(length).decode("utf-8", errors="replace")


def _fixed_size(value_type: int) -> int | None:
    if value_type in (0, 1, 7):
        return 1
    if value_type in (2, 3):
        return 2
    if 4 <= value_type <= 6:
        return 4
    if 10 <= value_type <= 12:
        return 8
    return None


def _skip(handle: BinaryIO, value_type: int) -> None:
    size = _fixed_size(value_type)
    if size is not None:
        handle.seek(size, 1)
        return
    if value_type == 8:
        length = _read("Q", handle)
        if length > MAX_STRING:
            raise ValueError("GGUF string is too large")
        handle.seek(length, 1)
        return
    if value_type == 9:
        element_type = _read("I", handle)
        count = _read("Q", handle)
        if count > MAX_ARRAY:
            raise ValueError("GGUF array is too large")
        fixed = _fixed_size(element_type)
        if fixed is not None:
            handle.seek(count * fixed, 1)
            return
        if element_type == 8:
            for _ in range(count):
                length = _read("Q", handle)
                if length > MAX_STRING:
                    raise ValueError("GGUF string is too large")
                handle.seek(length, 1)
            return
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def _integer(handle: BinaryIO, value_type: int) -> int | None:
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}
    fmt = formats.get(value_type)
    if not fmt:
        _skip(handle, value_type)
        return None
    return max(0, int(_read(fmt, handle)))


def read_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return result
            version = _read("I", handle)
            if version not in (2, 3):
                return result
            _read("Q", handle)
            metadata_count = _read("Q", handle)
            if metadata_count > 1_000_000:
                return result
            for _ in range(metadata_count):
                key = _string(handle)
                value_type = _read("I", handle)
                wanted = key == "general.architecture" or key.endswith((
                    ".block_count", ".context_length", ".embedding_length",
                    ".attention.head_count", ".attention.head_count_kv",
                ))
                if not wanted:
                    _skip(handle, value_type)
                    continue
                if key == "general.architecture":
                    if value_type == 8:
                        result["architecture"] = _string(handle)
                    else:
                        _skip(handle, value_type)
                    continue
                value = _integer(handle, value_type)
                if value is None:
                    continue
                if key.endswith(".block_count"):
                    result["layerCount"] = value
                elif key.endswith(".context_length"):
                    result["contextLength"] = value
                elif key.endswith(".embedding_length"):
                    result["embeddingLength"] = value
                elif key.endswith(".attention.head_count_kv"):
                    result["attentionHeadCountKv"] = value
                elif key.endswith(".attention.head_count"):
                    result["attentionHeadCount"] = value
    except (OSError, ValueError, struct.error):
        return {}
    return result


def has_nextn_tensors(path: Path) -> bool:
    """True when the GGUF carries NextN/MTP draft tensors (Qwen3.x-MTP class).

    Walks only the tensor directory — names, shapes, offsets — never the data
    section, so this stays cheap even on multi-gigabyte files.
    """
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return False
            version = _read("I", handle)
            if version not in (2, 3):
                return False
            tensor_count = _read("Q", handle)
            metadata_count = _read("Q", handle)
            if tensor_count > 1_000_000 or metadata_count > 1_000_000:
                return False
            for _ in range(metadata_count):
                _string(handle)
                _skip(handle, _read("I", handle))
            for _ in range(tensor_count):
                name = _string(handle)
                if "nextn" in name.lower():
                    return True
                dimensions = _read("I", handle)
                if dimensions > 8:
                    return False
                handle.seek(dimensions * 8 + 4 + 8, 1)
            return False
    except (OSError, ValueError, struct.error):
        return False
