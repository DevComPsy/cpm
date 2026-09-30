"""Combine the results of all tasks of the job array into one file."""

from pathlib import Path

import pandas as pd

files = sorted(Path("results").glob("fit_*.csv"))
fits = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
fits.to_csv("fits.csv", index=False)
print(f"combined {len(files)} files: {fits.ppt.nunique()} participants")
