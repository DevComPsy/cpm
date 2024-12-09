import psutil

import multiprocess as mp
import ipyparallel as ipp

import numpy as np
import os


def get_cpu_core_id(*args):
    """
    This function returns the ID of the CPU core the current process is running on.

    Returns
    -------
    int
        The ID of the CPU core.
    """
    import psutil
    import numpy as np
    import os

    # Get the current process
    process = psutil.Process(os.getpid())
    # Get the CPU affinity (list of cores the process can run on)
    cpu_affinity = process.cpu_affinity()
    # Get the current CPU core the process is running on
    current_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    # Find the core with the highest usage
    return np.array(np.asarray(current_cpu) > 0).sum()


def in_ipynb():
    """
    This function detects if the code is running in an ipython notebook or not.

    Returns
    -------
    bool
        True if the code is running in an ipython notebook, False otherwise
    """
    try:
        cfg = get_ipython().config
        return True
    except NameError:
        return False


def detect_parallel_method():
    """
    Detect the parallel execution method based on the environment.

    Returns
    -------
    str
        The detected parallel execution method.
    """
    if in_ipynb():
        return "ipyparallel"
    else:
        return "multiprocess"


def execute_parallel(job, data, method=None, cl=None):
    """
    Execute a job in parallel using the specified method.

    Parameters
    ----------
    job : function
        The job to execute.
    data : iterable
        The data to process.
    method : str, optional
        The parallel execution method. Options are 'ipyparallel' and 'multiprocess'.
        If None, the method is determined based on the environment.
    cl : int, optional
        The number of cores to use for parallel processing.

    Returns
    -------
    result
        The result of the parallel execution.
    """
    if method is None:
        method = detect_parallel_method()

    if method == "ipyparallel":
        cluster = ipp.Cluster(n=cl)  # Create a cluster with 'cl' cores
        rc = cluster.start_and_connect_sync()
        return rc[:].map_sync(job, data)
    elif method == "multiprocess":
        with mp.Pool(cl) as pool:
            return pool.map(job, data)
    else:
        raise ValueError(f"Unknown parallel execution method: {method}")


# Example usage
if __name__ == "__main__":
    tasks = [np.random.rand(1000) for _ in range(24)]
    results = execute_parallel(get_cpu_core_id, tasks)
    print(results)
