
import numpy as np
import os

def data2array(filepath):
    """
    data2array takes in the filepath of a 
    data file like `restaurant.csv` in 
    data directory, and returns a 1d array
    of data.

    :Example:
    >>> fp = os.path.join('data', 'restaurant.csv')
    >>> arr = data2array(fp)
    >>> isinstance(arr, np.ndarray)
    True
    >>> arr.dtype == np.dtype('float64')
    True
    >>> arr.shape[0]
    100000
    """
    fh = open(filepath)
    fh.readline()

    data = [float(line.strip()) for line in fh]

    return np.array(data)

def ends_in_9(arr):
    """
    ends_in_9 takes in an array of dollar amounts 
    and returns the proprtion of values that end 
    in 9 in the hundredths place.

    :Example:
    >>> arr = np.array([23.04, 45.00, 0.50, 0.09])
    >>> out = ends_in_9(arr)
    >>> 0 <= out <= 1
    True
    """
    rounded = np.round(arr * 100)
    return np.count_nonzero(rounded % 100 % 10 == 9) / len(rounded)
