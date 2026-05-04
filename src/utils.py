import json
from pathlib import Path
from typing import Any


class JsonError(Exception):
    ...


def read_json(path: str) -> Any:
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
