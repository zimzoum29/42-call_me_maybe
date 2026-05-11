"""Utility helpers for small JSON file I/O and errors.

This module provides a small `JsonError` exception used by the rest of
the package plus helpers to read and write JSON files with consistent
error translation.
"""

import json
from pathlib import Path
from typing import Any


class JsonError(Exception):
    """Raised for errors dealing with JSON input/output or validation."""


def read_json(path: str) -> Any:
    """Read a JSON file and return the parsed object.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        The parsed JSON structure (list/dict/etc.).

    Raises:
        JsonError: For file not found, permission errors or invalid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except FileNotFoundError as error:
        raise JsonError(f"Invalid path: '{path}'") from error
    except PermissionError as error:
        raise JsonError(f"Permission error at: '{path}'") from error
    except json.JSONDecodeError as error:
        raise JsonError(
            f"Invalid input file format, JSON format required: {path}"
        ) from error
    except OSError as error:
        raise JsonError(
            f"Unexpected error while reading '{path}': {error}"
        ) from error


def write_json(path: str, data: Any) -> None:
    """Write `data` as JSON to `path`, creating parent directories.

    The function ensures parent directories exist and writes UTF-8 JSON
    with a trailing newline.

    Args:
        path: Destination filesystem path.
        data: JSON-serializable object to write.

    Raises:
        JsonError: On permission or other OS-level write failures.
    """
    filepath = Path(path)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")
    except PermissionError as error:
        raise JsonError(f"Permission denied: '{path}'") from error
    except OSError as error:
        raise JsonError(
            f"Unexpected error while writing '{path}': {error}"
        ) from error
