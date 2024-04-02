"""
Tests for `cpm` module.
"""

# %%
import numpy as np
import pandas as pd
import time
import sys
from dask import dataframe as dd
from dask import array as da
from dask import delayed

# %%
x = dd.from_array(np.random.uniform(0, 1, (100, 5)))

x.compute()

# %%

y = dd.from_array(np.random.uniform(0, 1, (100, 5)))

y.compute()

# %%

one = dd.concat([y, x])

one.compute()

big = []
for k in range(100):
    ppt = []
    for j in range(100):
        k = np.random.uniform(0, 1, 5)
        ppt.append({"a": k, "b": k + 1, "c": k + 2})
    big.append(ppt)

###############################################################################
# %%


@delayed
def row(trial):
    element = 0
    for key, value in trial.items():
        if isinstance(value, int) or isinstance(value, float):
            value = np.array([value])
        if isinstance(value, np.ndarray):
            value = value.flatten()
        out = dd.from_array(np.array([value]))
        out.columns = [f"{key}_{i}" for i in range(out.shape[1])]
        if element == 0:
            row = out
            element += 1
        else:
            row = dd.concat([row, out], axis=1)
    return row


def convert(session):
    counts = 0
    jobs = []
    for trial in session:
        jobs.append(row(trial))
    return delayed(dd.concat)(jobs, axis=0)


# %%

start = time.time()
tracking = 0
for k in big:
    percentage = np.round(((tracking + 1) / len(big)) * 100)
    print(f"Percentage complete: {percentage} %")
    sys.stdout.flush()
    single = convert(k)
    if tracking > 0:
        all = dd.concat([all, single.compute()], axis=0)
    else:
        all = single.compute()
    tracking += 1

# %%

all.compute()

end = time.time()
print(f"{np.round((end - start)/60, 2)} minutes")

# %%

import dask


# %%
def ros(fun, pars):
    return fun(pars)


def inin(pars):
    return pars**2


out = []
for i in range(10):
    # pp = dask.delayed(float)(i)
    # tt = dask.delayed(inin)(i)
    ff = dask.delayed(ros)(inin, i)
    out.append(ff)

# %%
total = dask.delayed()(dask.compute)(out)
total.compute()

# %%

from multiprocess import Pool
import numpy as np


def f(x):
    return x * x * x**x


input = Pool(8).map(f, np.random.uniform(0, 10, (2000000, 10)))


# %%
