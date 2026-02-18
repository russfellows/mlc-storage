"""
Optional dependency detection for KV Cache Benchmark.

Centralizes try-import guards so other modules can check availability
without scattered try/except blocks.
"""

# Optional YAML support for config file loading
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

# Alias for backward compatibility
YAML_AVAILABLE = HAS_YAML

# Optional GPU libraries
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

TORCH_AVAILABLE = HAS_TORCH

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

CUPY_AVAILABLE = HAS_CUPY

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None
    HAS_TIKTOKEN = False

TIKTOKEN_AVAILABLE = HAS_TIKTOKEN

# Optional pandas/openpyxl for XLSX output
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

PANDAS_AVAILABLE = HAS_PANDAS

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    openpyxl = None
    HAS_OPENPYXL = False

OPENPYXL_AVAILABLE = HAS_OPENPYXL
