'''Define the settlement calculation based upon the trading amounts of the storage system'''

def settlementModel(storage_system_inst,energy,SP):
    '''
    Calculate the trading amounts for settlement at the end of the trading day.

    Parameters
    ----------
    system_assumptions : dictionary
        Dictionary of the assumed parameters for the system.
    energy : list
        List containing charged and discharged energy for the day.
    SP : list
        List containing spot prices for each trading interval in the day.

    Returns
    -------
    TA : list
        List containing the discharged and charged trading amount.

    '''
    
    # Define parameters
    dischargedEnergy = energy[0]
    chargedEnergy = energy[1]
    TA_ch = []
    TA_dis = []
    
    for s in range(0,48):
        for t in range(0,6):
            
            # Calculate the trade revenue for the dispatch interval
            TA_s = (storage_system_inst.dlf_gen*storage_system_inst.mlf_gen*dischargedEnergy[t+6*s]-storage_system_inst.dlf_load*storage_system_inst.mlf_load*chargedEnergy[t+6*s])*SP[s]
            if TA_s < 0:
                TA_ch.append(TA_s)
                TA_dis.append(0)
            else:
                TA_ch.append(0)
                TA_dis.append(TA_s)
    
    TA = [TA_dis,TA_ch]
    
    return TA