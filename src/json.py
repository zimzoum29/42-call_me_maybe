import json
from pathlib import Path
from typing import Any


class JsonError(Exception):
    ...


def read_json(path: str):
    filepath = Path(path)

    if not filepath.exists():
        raise JsonError(f"File not found: {path}")

    try:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except PermissionError as e:
        raise JsonError(f"Permission denied: {path}") from e
    except json.JSONDecodeError as e:
        raise JsonError(f"Invalid JSON format in file: '{path}'") from e

    except Exception as e:
        raise JsonError(f"Unexpected error while reading '{path}': {e}") from e


def write_json(path: str, data: Any) -> None:
    file_path = Path(path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except PermissionError as e:
        raise JsonError(f"Permission denied: '{path}'") from e

    except Exception as e:
        raise JsonError(f"Unexpected error while writing '{path}': {e}") from e
