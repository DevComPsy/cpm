## space bus is a stop-signal task, so we can actually include several different analyses methods
## Here is an incomplete list: integration method, correlation of go and no-go RTs, 

def preprocess(data=None, id=None, trialtypes=None, rt=None, accuracy=None, exclude_method=None):
    """
    Process spacebus data

    Examples
    --------
    >>> import cpm.brainexplorer as be
    >>> import pandas as pd
    >>> spacebus = pd.read_csv('spacebus.csv')
    >>> spacebus_aggregate = be.spacebus.preprocess(data=spacebus, id='userID', trialtypes='stopSignal', rt='RT_seconds', accuracy)
    """
    pass