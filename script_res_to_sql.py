""" Module for combining all results into a single sqlite database. """

import sqlite3
from glob import glob

import pandas as pd

all_data = pd.concat([
    pd.read_parquet(f) for f in sorted(glob("results/all/*.parquet.gz"))
])
all_data.loc[all_data["entropy"].isna(), "entropy"] = 1.0

conn = sqlite3.connect("results/all_res.db")
all_data.to_sql("results", conn, if_exists="replace", index=False)
conn.close()
print("Success")
