"""
New hierarchical routing implementation for the Grace project (multi-main flows).

This package keeps the v2 code separate from the legacy implementation so that
experiments can run side-by-side without modifying the original modules.
"""

import os
import sys

# Ensure ns.py is importable for submodules that rely on the simulator package.
_ns_path = os.path.join(os.getcwd(), "ns.py")
if _ns_path not in sys.path:
    sys.path.append(_ns_path)
