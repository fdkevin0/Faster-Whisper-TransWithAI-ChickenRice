#!/usr/bin/env python3
"""
Standalone XPU inference script.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from faster_whisper_transwithai_chickenrice_xpu.infer import main


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(__file__))
    main()
