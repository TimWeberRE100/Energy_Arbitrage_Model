'''
Assist in debugging the software.

Functions
---------
debugStorage
'''

from pprint import pprint

def debugStorage(storage_system_inst):
    '''
    Calculate the open-circuit voltage of the cells at a particular SOC.

    Parameters
    ----------
    storage_system_inst : storage_system
        Object containing the storage system attributes and current state.

    Returns
    -------
    None.

    Side-effects
    ------------
    Print the list of attributes in the storage_system_inst object and the objects from which it is constructed.
    '''

    pprint(vars(storage_system_inst))
    
    try:
        for obj in storage_system_inst._storage_system__objects:
            pprint(vars(obj))
    except:
        pass