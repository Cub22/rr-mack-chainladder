"""Make ``src/`` importable when the package has not been installed.

``pip install -e .`` is the intended way to run the suite; this keeps a bare
``pytest tests`` working from a fresh checkout too.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
