import multiprocess as mp
import ipyparallel as ipp
import dill

__all__ = ["detect_cores", "execute_parallel", "detect_parallel_method", "in_ipynb"]


def ipyparallel_pandas_to_list(dataframe):
    """
    Convert a pandas DataFrame to a list of dictionaries.

    Parameters
    ----------
    dataframe : pandas.DataFrame.groupby
        The grouped DataFrame to convert.

    Returns
    -------
    list
        A list of tuples, where each element has the:
        - key: the group key
        - value: the group data as pandas DataFrame
    """
    output = []
    for key, value in dataframe:
        output.append((key, value))
    return output


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


def detect_cores():
    """
    Detect the number of cores available for parallel processing.

    Returns
    -------
    int
        The number of cores available for parallel processing.
    """
    return mp.cpu_count()


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


def execute_parallel(
    job, data, method=None, cl=None, pandas=True, libraries=["numpy", "pandas"]
):
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
    libraries : list, optional
        A list of modules to import before executing the job. The name must correspond to the module name as it was imported in the main script.

    Returns
    -------
    result
        The result of the parallel execution.
    """
    if pandas:
        data = ipyparallel_pandas_to_list(data)

    if method is None:
        method = detect_parallel_method()
        print(f"Detected parallel execution method: {method}")

    if method == "ipyparallel":
        cluster = ipp.Cluster(n=cl)  # Create a cluster with 'cl' cores
        rc = cluster.start_and_connect_sync()
        rc.wait_for_engines(n=cl)
        rc[:].use_dill()

        @ipp.require(*libraries)
        def wrapped_job(*args, **kwargs):
            for lib in libraries:
                exec(f"import {lib}")
            return job(*args, **kwargs)

        return rc[:].map_sync(wrapped_job, data)
    elif method == "multiprocess":
        with mp.Pool(cl) as pool:
            return pool.map(job, data)
    else:
        raise ValueError(f"Unknown parallel execution method: {method}")
