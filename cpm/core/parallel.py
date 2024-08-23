import multiprocess as mp
import ipyparallel as ipp
from mpi4py import MPI

import numpy as np
import os


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


def in_slurm():
    """
    This function detects if the code is running in a slurm job or not.

    Returns
    -------
    bool
        True if the code is running in a slurm job, False otherwise.
    """
    return "SLURM_JOB_ID" in os.environ


def detect_cores(backend="multiprocess"):
    """
    This function detects the number of cores available for parallel processing.

    Parameters
    ----------
    backend : str
        The backend to use for parallel processing. Options are 'multiprocess' and 'ipyparallel'.

    Returns
    -------
    int
        The number of cores available for parallel processing.
    """
    if backend == "multiprocess":
        return mp.cpu_count()
    elif backend == "ipyparallel":
        if in_ipynb():
            try:
                cluster = ipp.Cluster(
                    n=4
                )  # Create a cluster with 4 engines (adjust the number as needed)
                cluster.start_and_connect_sync()
                rc = cluster.client
            except:
                return 1
        else:
            return 1
    elif backend == "mpi":
        comm = MPI.COMM_WORLD
        return comm.Get_size()
    else:
        return 1


def detect_mpi():
    """
    This function detects if MPI is available for parallel processing.

    Returns
    -------
    bool
        True if MPI is available, False otherwise.
    """
    return MPI.COMM_WORLD.Get_size() > 1


def detect_backend():
    """
    This function detects the backend to use for parallel processing.

    Returns
    -------
    str
        The backend to use for parallel processing. Options are 'multiprocess', 'ipyparallel', 'mpi', and 'serial'.
    """
    if detect_mpi():
        return "mpi"
    elif detect_cores("ipyparallel") > 1:
        return "ipyparallel"
    elif detect_cores("multiprocess") > 1:
        return "multiprocess"
    else:
        return "serial"


def parallel_map(func, iterable, backend=None, cl=None):
    """
    This function applies a function to each element of an iterable in parallel.

    Parameters
    ----------
    func : function
        The function to apply to each element of the iterable.
    iterable : iterable
        The iterable to apply the function to.
    backend : str
        The backend to use for parallel processing. Options are 'multiprocess', 'ipyparallel', 'mpi', and 'serial'.
    cl : int
        The number of cores to use for parallel processing.

    Returns
    -------
    list
        The list of results from applying the function to each element of the iterable.
    """
    if backend is None:
        backend = detect_backend()

    if backend == "multiprocess":
        with mp.Pool(cl) as pool:
            return pool.map(func, iterable)
    elif backend == "ipyparallel":
        if in_ipynb():
            cluster = ipp.Cluster(n=cl)  # Create a cluster with 'cl' cores
            cluster.start_and_connect_sync()
            rc = cluster.client
            return rc[:].map_sync(func, iterable)
    elif backend == "mpi":
        ## divide the work between nodes and cores
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        n = len(iterable)

        ## calculate the number of elements to process
        chunk = n // size
        start = rank * chunk
        end = start + chunk
        if rank == size - 1:
            end = n

        ## process the elements
        results = []
        for i in range(start, end):
            results.append(func(iterable[i]))

        ## gather the results
        results = comm.gather(results, root=0)

        ## return the results
        if rank == 0:
            return [item for sublist in results for item in sublist]
        else:
            return None
    elif backend == "serial":
        return list(map(func, iterable))
    else:
        return [func(i) for i in iterable]
