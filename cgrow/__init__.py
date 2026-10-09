"""cgrow -- contracted growth.

Interpreter guard. Twice in this project a background job silently ran under
a conda *base* interpreter instead of the project environment (different torch /
sklearn / numpy), producing numbers that were not comparable with the frozen
evaluation protocol and had to be discarded. Every experiment imports this
package, so the check lives here. It refuses only a conda base interpreter
rather than requiring one specific environment, so any virtualenv or conda env
works. Override deliberately with CGROW_ALLOW_ANY_PYTHON=1.
"""
import os as _os
import sys as _sys

_exe = _sys.executable.replace("\\", "/").lower()
if ("anaconda3/python" in _exe or "miniconda3/python" in _exe) \
        and not _os.environ.get("CGROW_ALLOW_ANY_PYTHON"):
    raise RuntimeError(
        f"cgrow: refusing to run under the conda BASE interpreter "
        f"({_sys.executable}). Use a dedicated environment "
        f"(python -m venv .venv, then pip install -r requirements.txt), "
        f"or set CGROW_ALLOW_ANY_PYTHON=1 to override deliberately.")
