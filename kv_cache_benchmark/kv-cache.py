#!/usr/bin/env python3
"""
KV Cache Benchmark - Multi-Tier Performance Comparison
Kingston Digital, 2025
Licensed under the Apache License, Version 2.0 (the "License")
MLPerf Storage Working Group

Thin wrapper around the kv_cache package (modular_architecture/kv_cache/).
All implementation has been refactored into the package while this file
preserves backward compatibility for existing scripts and test imports.
"""

import sys
import os

# Add the script's directory to sys.path so `import kv_cache` resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-export all public symbols for backward compatibility with test imports
from kv_cache import *  # noqa: F401,F403
from kv_cache.cli import main

if __name__ == '__main__':
    main()
