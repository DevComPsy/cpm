# https://berkeley-scf.github.io/tutorial-dask-future/python-dask.html#5-using-different-schedulers
import time
import numpy as np


def ackley(x):
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2.0 * np.pi * x[0]) + np.cos(2.0 * np.pi * x[1]))
    return -20.0 * np.exp(arg1) - np.exp(arg2) + 20.0 + np.e


parameters = np.random.uniform(-32.768, 32.768, (100000, 2))

start = time.time()
storage = []
for input in parameters:
    storage.append(ackley(input))
end = time.time()

print("Time taken for serial execution: ", end - start)

import dask
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=4)
c = Client(cluster)
parameters_scattered = c.scatter(parameters)

p = 100000

job = c.submit(ackley, parameters_scattered)  # 0.0002 sec.
c.compute(job)  # 3.4 sec.|

futures = [dask.delayed(ackley)(input) for input in parameters]
t0 = time.time()
results = dask.compute(futures)
time.time() - t0  # 3.4 sec.
