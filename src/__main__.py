"""Primary module entrypoint for the package.

This module allows running the package with `python -m src` and
delegates to `src.main.main()`.
"""

from src.main import main

if __name__ == "__main__":
    main()
