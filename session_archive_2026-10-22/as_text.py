"""Run an evaluation script with the label columns read as text, as they are when the full test set (which has date
labels) is read. Usage: python as_text.py <script.py> [args...]"""
import runpy, sys
import pandas as pd
_read_csv = pd.read_csv
def read_csv(*a, **k):
    k.setdefault("dtype", {"Ground Truth Answer": str, "Lower Limit": str, "Upper Limit": str})
    return _read_csv(*a, **k)
pd.read_csv = read_csv
import os
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
