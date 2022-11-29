from pprint import pprint

def debugStorage(storage_system_inst):
    pprint(vars(storage_system_inst))
    
    try:
        for obj in storage_system_inst._storage_system__objects:
            pprint(vars(obj))
    except:
        pass