#!/usr/bin/env python3
"""Main entry point for the Text Detection App."""

import sys
from pathlib import Path

# Add src to path so the package can be imported when run directly
project_root = Path(__file__).parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from text_detector.__main__ import main

if __name__ == "__main__":
    main()
