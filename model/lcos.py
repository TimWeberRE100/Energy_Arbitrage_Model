'''
Calculate the levelised cost of storage metrics of the system over the lifetime.

Functions
---------
EOL_LCOS

EOL_LCOS_Deg
'''

def EOL_LCOS(annualDischargedEnergy,annual_TA_dis,annual_TA_ch,storage_system_inst,Year):
    '''
    Calculate the LCOS metrics at the end of the year for the non-degrading storage system (e.g. PHS).

    Parameters
    ----------
    annualDischargedEnergy : list
        List of daily energy discharged by the system in the year.
    annual_TA_dis : list
        List of daily trading amounts from discharging by the system in the year.
    annual_TA_ch : list
        List of daily trading amounts from charging by the system in the year.
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.
    Year : integer
        Year from which prices are used.

    Returns
    -------
    RADP : float
        Long-run marginal costs of the system (required average discharge price)
    AADP : float
        Long-run marginal revenue of the system (available average discharge price)

    Side-effects
    ------------
    None.
    '''
        
    # Define annual parameter lists
    taxFactor_num = []
    RADP_num = []
    RADP_den = []
    VOM_y = []
    AADP_num = []
    AADP_den = []
    
    # Calculate total OCC
    CAPEX = storage_system_inst.OCC_p*storage_system_inst.power_capacity + storage_system_inst.OCC_e*storage_system_inst.energy_capacity + storage_system_inst.OCC_f
    baseValue = CAPEX
    
    # Calculate annual parameters
    for y in range(1,storage_system_inst.lifetime+1):
        
        # Define number of days in year
        if Year % 4 == 0:
            total_days = 366
        else:
            total_days = 365
        
        d_y = (baseValue*(total_days/365)*(2/storage_system_inst.lifetime))/CAPEX
        baseValue -= baseValue*(total_days/365)*(2/storage_system_inst.lifetime)
        
        taxFactor_num.append(d_y*(1+storage_system_inst.discountRate)**(-y))
        RADP_num.append((storage_system_inst.FOM*storage_system_inst.power_capacity*1000+sum(annual_TA_ch))*(1+storage_system_inst.discountRate)**(-y))
        RADP_den.append(sum(annualDischargedEnergy)*(1+storage_system_inst.discountRate)**(-y))
        VOM_y.append(storage_system_inst.VOMd*(1+storage_system_inst.discountRate)**(-y))
        AADP_num.append(sum(annual_TA_dis)*(1+storage_system_inst.discountRate)**(-y))
        AADP_den.append(sum(annualDischargedEnergy)*(1+storage_system_inst.discountRate)**(-y))
    
    # Calculate LCOS parameters
    taxFactor =(1-storage_system_inst.i_credit-storage_system_inst.corp_tax*(1-storage_system_inst.i_credit)*sum(taxFactor_num))/(1-storage_system_inst.corp_tax)
    RADP = (CAPEX*taxFactor + sum(RADP_num))/sum(RADP_den)+sum(VOM_y)
    AADP = sum(AADP_num)/sum(AADP_den)
    
    return [RADP,AADP]

def EOL_LCOS_Deg(simDischargedEnergy,sim_TA_dis,sim_TA_ch,Year,storage_system_inst):
    '''
    Calculate the LCOS metrics at the end of the simulation for the degrading storage system (e.g. BESS).

    Parameters
    ----------
    simDischargedEnergy : float
        Total discharged energy over the lifetime.
    sim_TA_dis : float
        Total trading amount from discharging over the system lifetime.
    sim_TA_ch : float
        Total trading amount from charging over the system lifetime.
    Year : integer
        Year from which prices are used.
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.

    Returns
    -------
    RADP : float
        Long-run marginal costs of the system (required average discharge price)
    AADP : float
        Long-run marginal revenue of the system (available average discharge price)

    Side-effects
    ------------
    None.
    '''
    
    # Define annual parameter lists
    taxFactor_num = []
    RADP_num = []
    RADP_den = []
    VOM_y = []
    AADP_num = []
    AADP_den = []
    
    # Calculate total OCC
    CAPEX = storage_system_inst.OCC_p*storage_system_inst.power_capacity + storage_system_inst.OCC_e*storage_system_inst.energy_capacity + storage_system_inst.OCC_f
    baseValue = CAPEX
    
    # Calculate annual parameters
    for y in range(1,storage_system_inst.lifetime+1):
        # Define number of days in year
        if Year % 4 == 0:
            total_days = 366
        else:
            total_days = 365
        
        d_y = (baseValue*(total_days/365)*(2/storage_system_inst.lifetime))/CAPEX
        baseValue -= baseValue*(total_days/365)*(2/storage_system_inst.lifetime)
        
        taxFactor_num.append(d_y*(1+storage_system_inst.discountRate)**(-y))
        RADP_num.append((storage_system_inst.FOM*storage_system_inst.power_capacity*1000+sim_TA_ch[y-1])*(1+storage_system_inst.discountRate)**(-y))
        RADP_den.append(simDischargedEnergy[y-1]*(1+storage_system_inst.discountRate)**(-y))
        VOM_y.append(storage_system_inst.VOMd*(1+storage_system_inst.discountRate)**(-y))
        AADP_num.append(sim_TA_dis[y-1]*(1+storage_system_inst.discountRate)**(-y))
        AADP_den.append(simDischargedEnergy[y-1]*(1+storage_system_inst.discountRate)**(-y))
    
    # Calculate LCOS parameters
    taxFactor =(1-storage_system_inst.i_credit-storage_system_inst.corp_tax*(1-storage_system_inst.i_credit)*sum(taxFactor_num))/(1-storage_system_inst.corp_tax)
    RADP = (CAPEX*taxFactor + sum(RADP_num))/sum(RADP_den)+sum(VOM_y)
    AADP = sum(AADP_num)/sum(AADP_den)
    
    return [RADP,AADP]