import multiprocess as mp
import ipyparallel as ipp
import os
import socket

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


def detect_nodes():
    """
    This function detects the number of nodes in the current slurm job.

    Returns
    -------
    int
        The number of nodes in the current slurm job.
    """
    return int(os.environ.get("SLURM_JOB_NUM_NODES", 1))


def execute_ipyparallel(job, data):
    """
    This function executes a job in parallel using ipyparallel.

    Parameters
    ----------
    job : function
        The function to be executed in parallel.
    data : list
        The list of data to be processed in parallel.

    Returns
    -------
    list
        The result of the job executed in parallel.
    """
    if not in_ipynb():
        raise Exception("This function should only be called from an ipython notebook.")

    c = ipp.Client()
    dview = c[:]
    dview.block = True
    dview.push({"data": data})
    dview.push({"job": job})
    dview.execute("result = [job(d) for d in data]")
    result = dview.pull("result", block=True)
    return result


def execute_multiprocess(job, data):
    """
    This function executes a job in parallel using multiprocess.

    Parameters
    ----------
    job : function
        The function to be executed in parallel.
    data : list
        The list of data to be processed in parallel.

    Returns
    -------
    list
        The result of the job executed in parallel.
    """
    if in_ipynb():
        raise Exception("This function should not be called from an ipython notebook.")

    with mp.Pool() as pool:
        result = pool.map(job, data)
    return result


def execute_serial(job, data):
    """
    This function executes a job in serial.

    Parameters
    ----------
    job : function
        The function to be executed in serial.
    data : list
        The list of data to be processed in serial.

    Returns
    -------
    list
        The result of the job executed in serial.
    """
    results = list(map(job, data))
    return results


def distribute_across_nodes(job, data):
    """
    This function distributes a job across multiple nodes in a slurm job.

    Parameters
    ----------
    job : function
        The function to be executed in parallel.
    data : list or pandas.DataFrame.groupby()
        The data to be processed in parallel.

    Returns
    -------
    list
        The result of the job executed in parallel.
    """
    if not in_slurm():
        raise Exception("This function should only be called from a slurm job.")

    if detect_nodes() > 1:
        return execute_multiprocess(job, data)

    number_of_nodes = detect_nodes()
    # TODO: Implement this function
    # TODO: when pandas, distribute different then when list
