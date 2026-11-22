"""Which test set the scripts in this folder build.

Default: the current set (labels of dataset/test_data_10_22_corrected.csv); results are written into this folder.
MEDCALC_VARIANT=astra: the same set with the 45 rows changed by the unreviewed Astra corrections taken from the Astra
re-run (extra_runs/astra_corrections_rerun_2026-10-22, 10-22 14:24): their notes, Relevant Entities, labels and
explanations, their outputs (all 8 models, every style, code prompt) and their error classifications. Results are
written into astra/ (same layout as this folder).

  MEDCALC_VARIANT=astra python3 consolidate.py   (and so on, in the order of README.md)"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root

ASTRA = os.environ.get("MEDCALC_VARIANT", "") == "astra"
OUT = f"{HERE}/astra" if ASTRA else HERE  # where outputs/, tables/ and the CSVs go

ASTRA_RUN = f"{ROOT}/extra_runs/astra_corrections_rerun_2026-10-22"
ASTRA_TEST = f"{ASTRA_RUN}/input/test_data_10_22_astra.csv"  # the test set the Astra re-run used
ASTRA_ROWS = set(pd.read_csv(f"{ASTRA_RUN}/astra_changes_applied.csv", dtype=str)["Row Number"]) if ASTRA else set()
# the columns the Astra corrections change (they change nothing outside ASTRA_ROWS)
ASTRA_COLUMNS = ["Patient Note", "Relevant Entities", "Ground Truth Answer", "Lower Limit", "Upper Limit",
                 "Ground Truth Explanation"]


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("Row Number")


def with_astra(table):
    """table (indexed by Row Number) with the Astra rows' changed columns taken from the Astra re-run's test set."""
    if ASTRA:
        rows = sorted(ASTRA_ROWS, key=int)
        table.loc[rows, ASTRA_COLUMNS] = read(ASTRA_TEST).loc[rows, ASTRA_COLUMNS]
    return table


def labels():
    """Labels every output is scored against, indexed by Row Number."""
    return with_astra(read(f"{ROOT}/dataset/test_data_10_22_corrected.csv"))
