## Python coding style (Tree-Seg)

This repository follows the **Google Python Style Guide**. The goal is
consistent, readable, and maintainable code.

### General rules

- **Avoid Jupyter notebooks for main project code**: notebooks are fine for
  quick experiments, but commit reusable logic as regular Python modules.
- **Documentation must be correct**: keep docstrings and README instructions
  accurate and up to date; run instructions end-to-end once after writing them.
- **Docstrings should be Sphinx-friendly**: prefer Google-style docstrings and
  keep them easy to render (no broken indentation / markup).

### Formatting

- **Max line length**: **100 characters**.
- **Strings**: use **double quotes** (`"..."`) by default.
- **String formatting**: prefer **f-strings**. For `logging`, lazy formatting is
  acceptable when appropriate (e.g. `logger.info("x=%s", x)`).

### Imports

- **3 import blocks**, separated by a blank line:
  - **standard library**
  - **third-party**
  - **local (this repo)**
- **Alphabetical order** within each block.

Example:

```python
import json
from pathlib import Path

import numpy as np

from lib.projection_equirect import world_point_to_uv_equirect
```

### TODO / FIXME tagging

- **Always tag an owner** using initials:
  - `# TODO(YJC): ...`
  - `# FIXME(YJC): ...`

### Attribution

- **If you copy/adapt code from open source**, add a short comment with the
  source (project name + file/path and commit/tag if possible).


