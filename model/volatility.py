'''
Function that calculates the volatility (standard deviation) of a list of prices.
'''

import numpy as np

def volatility(price_list):
    """
    Calculates the volatility of a list of energy prices.
    
    Parameters
    ----------
    price_list : list
        List of energy prices in [$/MWh].

    Returns
    -------
    price_vol : float
        Standard deviation of the energy prices in price_list.

    """
    price_vol = np.std(price_list)
    return price_vol