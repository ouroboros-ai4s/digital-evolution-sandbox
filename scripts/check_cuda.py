#!/usr/bin/env python
"""Quick environment diagnostic: confirm the basic conda env sees the GPU.

    D:/anaconda3/envs/basic/python.exe scripts/check_cuda.py

NOTE: a bare `python` on PATH resolves to a DIFFERENT cpu-only torch 2.5.1 and
will print False — always invoke the explicit basic python (torch 2.10+cu128).
"""
import torch

print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device      : {torch.cuda.get_device_name(0)}")
