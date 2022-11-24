def EOL_LCOS(annualDischargedEnergy,annualChargedEnergy,annual_TA_dis,annual_TA_ch,system_assumptions,Year):
    '''
    Calculate the LCOS metrics at the end of the year for the PHS.

    Parameters
    ----------
    annualDischargedEnergy : list
        List of daily energy discharged by the system in the year.
    annualChargedEnergy : list
        List of daily energy charged by the system in the year.
    annual_TA_dis : list
        List of daily trading amounts from discharging by the system in the year.
    annual_TA_ch : list
        List of daily trading amounts from charging by the system in the year.
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.
    Year : integer
        Year from which prices are used.

    Returns
    -------
    list
        List containing RADP and AADP.

    '''
    
    discountRate = float(system_assumptions["discountRate"])
    i = float(system_assumptions["i"])
    alpha = float(system_assumptions["alpha"])
    OCCp = float(system_assumptions["OvernightCapitalCost (power)"])
    OCCe = float(system_assumptions["OvernightCapitalCost (energy)"])
    FOM = float(system_assumptions["FOM"])
    VOMd = float(system_assumptions["VOMd"])
    lifetime = int(system_assumptions["Lifetime"])
    C_p = int(system_assumptions["power_capacity"])
    C_e = int(system_assumptions["energy_capacity"])
    OCC_other = int(system_assumptions["OtherOCC"])
    
    # Define annual parameter lists
    taxFactor_num = []
    RADP_num = []
    RADP_den = []
    VOM_y = []
    AADP_num = []
    AADP_den = []
    
    # Calculate total OCC
    CAPEX = OCCp*C_p + OCCe*C_e + OCC_other
    baseValue = CAPEX
    
    # Calculate annual parameters
    for y in range(1,lifetime+1):
        
        # Define number of days in year
        if Year % 4 == 0:
            total_days = 366
        else:
            total_days = 365
        
        d_y = (baseValue*(total_days/365)*(2/lifetime))/CAPEX
        baseValue -= baseValue*(total_days/365)*(2/lifetime)
        
        taxFactor_num.append(d_y*(1+discountRate)**(-y))
        RADP_num.append((FOM*C_p*1000+sum(annual_TA_ch))*(1+discountRate)**(-y))
        RADP_den.append(sum(annualDischargedEnergy)*(1+discountRate)**(-y))
        VOM_y.append(VOMd*(1+discountRate)**(-y))
        AADP_num.append(sum(annual_TA_dis)*(1+discountRate)**(-y))
        AADP_den.append(sum(annualDischargedEnergy)*(1+discountRate)**(-y))
    
    # Calculate LCOS parameters
    taxFactor =(1-i-alpha*(1-i)*sum(taxFactor_num))/(1-alpha)
    RADP = (CAPEX*taxFactor + sum(RADP_num))/sum(RADP_den)+sum(VOM_y)
    AADP = sum(AADP_num)/sum(AADP_den)
    
    return [RADP,AADP]

def EOL_LCOS_Deg(simDischargedEnergy,simChargedEnergy,sim_TA_dis,sim_TA_ch,system_assumptions,Year,degredationLife):
    '''
    Calculate the LCOS metrics at the end of the year for the BESS.

    Parameters
    ----------
    simDischargedEnergy : float
        Total discharged energy over the lifetime.
    simChargedEnergy : float
        Total charged energy over the lifetime.
    sim_TA_dis : float
        Total trading amount from discharging over the system lifetime.
    sim_TA_ch : float
        Total trading amount from charging over the system lifetime.
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.
    Year : integer
        Year from which prices are used.
    degredationLife : integer
        Lifetime of the system in years assumed for calculating asset degradation tax deduction.

    Returns
    -------
    list
        List containing RADP and AADP.

    '''
    
    # Define parameters
    discountRate = float(system_assumptions["discountRate"])
    i = float(system_assumptions["i"])
    alpha = float(system_assumptions["alpha"])
    OCCp = float(system_assumptions["OvernightCapitalCost (power)"])
    OCCe = float(system_assumptions["OvernightCapitalCost (energy)"])
    FOM = float(system_assumptions["FOM"])
    VOMd = float(system_assumptions["VOMd"])
    lifetime = degredationLife
    C_p = int(system_assumptions["power_capacity"])
    C_e = int(system_assumptions["energy_capacity"])
    OCC_other = int(system_assumptions["OtherOCC"])
    
    # Define annual parameter lists
    taxFactor_num = []
    RADP_num = []
    RADP_den = []
    VOM_y = []
    AADP_num = []
    AADP_den = []
    
    # Calculate total OCC
    CAPEX = OCCp*C_p + OCCe*C_e + OCC_other
    baseValue = CAPEX
    
    # Calculate annual parameters
    for y in range(1,lifetime+1):
        # Define number of days in year
        if Year % 4 == 0:
            total_days = 366
        else:
            total_days = 365
        
        d_y = (baseValue*(total_days/365)*(2/lifetime))/CAPEX
        baseValue -= baseValue*(total_days/365)*(2/lifetime)
        
        taxFactor_num.append(d_y*(1+discountRate)**(-y))
        RADP_num.append((FOM*C_p*1000+sim_TA_ch[y-1])*(1+discountRate)**(-y))
        RADP_den.append(simDischargedEnergy[y-1]*(1+discountRate)**(-y))
        VOM_y.append(VOMd*(1+discountRate)**(-y))
        AADP_num.append(sim_TA_dis[y-1]*(1+discountRate)**(-y))
        AADP_den.append(simDischargedEnergy[y-1]*(1+discountRate)**(-y))
    
    # Calculate LCOS parameters
    taxFactor =(1-i-alpha*(1-i)*sum(taxFactor_num))/(1-alpha)
    RADP = (CAPEX*taxFactor + sum(RADP_num))/sum(RADP_den)+sum(VOM_y)
    AADP = sum(AADP_num)/sum(AADP_den)
    
    return [RADP,AADP]