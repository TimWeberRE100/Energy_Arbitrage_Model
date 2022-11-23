# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:15:29 2021

@author: u5641776
"""

import pandas as pd
import pyomo.environ as pyo
import numpy as np
import concurrent.futures
import os
from pyomo.opt.results import SolverStatus

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

def Pmax_ij_aged(SOC,CE_batt,Ucell_nom,eff_sys,Ubatt_nom,Ubatt_SOC,R_cell,CE_cell):
    '''
    Calculates the maximum power limit for a particular state of charge, accounting for efficiency fade.

    Parameters
    ----------
    SOC : float
        State of charge of the system.
    CE_batt : integer
        Energy capacity [MWh] of the battery.
    Ucell_nom : float
        Nominal open-circuit voltage of the cells.
    eff_sys : float
        Efficiency of the system constituting auxiliary losses.
    Ubatt_nom : float
        Nominal open-circuit voltage of the battery.
    Ubatt_SOC : float
        Open-circuit voltage of the battery at the specified SOC.
    R_cell : float
        Internal resistance of the cell [Ohms].
    CE_cell : float
        Energy capacity [MWh] of the cell.

    Returns
    -------
    Pmax_current : float
        Maximum power limit ensuring efficiency losses > 80%.

    '''
    numerator = ((eff_sys**2)/4) - (0.8 - 0.5*eff_sys)**2
    denominator = eff_sys*((Ubatt_nom/Ubatt_SOC)**2)*R_cell*CE_cell
    coefficient = SOC*CE_batt*Ucell_nom
    
    Pmax_current = coefficient*(numerator/denominator)
    
    return Pmax_current

def Ploss_m_aged(power,CE_batt,Ucell_nom,eff_sys,R_cell,CE_cell):
    '''
    Calculates the power loss for a particular dispatch power, accounting for efficiency fade.

    Parameters
    ----------
    power : integer
        Dispatch power of the system [MW].
    CE_batt : integer
        Energy capacity [MWh] of the battery.
    Ucell_nom : float
        Nominal open-circuit voltage of the cells.
    eff_sys : float
        Efficiency of the system constituting auxiliary losses.
    R_cell : float
        Internal resistance of the cell [Ohms].
    CE_cell : float
        Energy capacity [Wh] of the cell.

    Returns
    -------
    Ploss_current : float
        Power loss at the specified dispatch power.

    '''
    numerator = power*R_cell*CE_cell
    denominator = eff_sys*0.5*CE_batt*Ucell_nom
    radicand = 0.25 - numerator / denominator
    
    if radicand < 0:
        eff_volt = 0.5
    else:
        eff_volt = 0.5 + np.sqrt(radicand)
    
    Ploss_current = (1-eff_volt*eff_sys)*power/(eff_volt*eff_sys)
    
    return Ploss_current

def min_greaterThan(load_PB,SP_s,risk_level):
    '''
    Calculates the minimum load price band that is greater than the specified spot price, accounting
    for the level of risk hedging.

    Parameters
    ----------
    load_PB : list
        The ordered list of up to 10 load price bands from smallest to largest [$/MWh].
    SP_s : float
        The forecast spot price for a trading interval [$/MWh].
    risk_level : integer
        Level of risk hedging (0 to 9), with lower lisk levels constituting price bands closer to
        the spot price.

    Returns
    -------
    float
        The price band used for the load bid in the trading interval.

    '''
    for b in range(0,len(load_PB)):
        if (load_PB[b] > SP_s) and (b + risk_level <= 9):
            return load_PB[b+risk_level]
    return load_PB[-1]

def max_lessThan(gen_PB_reverse,SP_s,risk_level):
    '''
    Calculates the maximum generator price band that is lower than the specified spot price, accounting
    for the level of risk hedging.

    Parameters
    ----------
    gen_PB_reverse : list
        The ordered list of up to 10 generator price bands [$/MWh] from largest to smallest.
    SP_s : float
        The forecast spot price for a trading interval [$/MWh].
    risk_level : integer
        Level of risk hedging (0 to 9), with lower lisk levels constituting price bands closer to
        the spot price.

    Returns
    -------
    float
        The price band used for the generator offer in the trading interval.

    '''
    for o in range(0,len(gen_PB_reverse)):
        if (gen_PB_reverse[o] < SP_s) and (o + risk_level <= 9):
            return gen_PB_reverse[o+risk_level]
    return gen_PB_reverse[-1]

def linearParameterDF(linearisation_df, system_type, parameterName):
    '''
    Builds a dataframe for a particular linearisation parameter

    Parameters
    ----------
    linearisation_df : DataFrame
        Dataframe of all linearisation parameters.
    system_type : string
        The name for the type of storage system.
    parameterName : string
        The name of the particular linearisation parameter within the linearisation_df.

    Returns
    -------
    parameterValue_df : DataFrame
        Dataframe containing the values for the particular linearisation parameter.

    '''
    parameterValue_df = linearisation_df.loc[(linearisation_df['Variable Name'] == parameterName) & (linearisation_df['System Type'] == system_type)]['Variable Value'].values   
    return parameterValue_df

def U_OCV(SOC):
    '''
    Calculates the open-circuit voltage of the cells at a particular SOC.

    Parameters
    ----------
    SOC : float
        State of charge of the system.

    Returns
    -------
    U_cell : float
        Cell open-circuit voltage.

    '''
    n1 = 5.163*np.exp(-(((SOC-1.794)/1.665)**2))
    n2 = 0.3296*np.exp(-(((SOC-0.6405)/0.3274)**2))
    n3 = 1.59*np.exp(-(((SOC-0.06475)/0.4406)**2))
    n4 = 5.184*np.exp(-(((SOC-(-0.531))/0.3059)**2))
    U_cell = sum([n1,n2,n3,n4])
    return U_cell

def R_cell(year_count,day,dispatch_interval,Temp,R_cell_initial,current_state):
    '''
    Calculates the internal resistance of the cells based on an empirical model.

    Parameters
    ----------
    year_count : integer
        Current year iteration of the simulation.
    day : integer
        Current day in the year.
    dispatch_interval : integer
        Current dispatch interval in the day.
    Temp : integer
        Temperature of the cells.
    R_cell_initial : float
        Internal resistance of the cells upon initializing the simulation.
    current_state : dictionary
        Dictionary of variables representing the current state of the system.

    Returns
    -------
    R_cell : float
        Updated internal resistance of the cells at the end of the dispatch interval.

    '''
    
    # Calculate the number of months that have been simulated
    ST = (365*(year_count)+day+dispatch_interval/288)/30
    
    # Calculate the average SOC during the simulation
    if year_count+day+dispatch_interval == 0:
        SOC_avg = 0.5
    else:
        SOC_avg = (current_state["SOC_sum"]) / (current_state["dispatch_intervals"])
        
    # Calculate the internal resistance
    R_cell = R_cell_initial + R_cell_initial*(6.9656*(10**(-8))*np.exp(0.05022*Temp)) * (2.897*np.exp(0.006614*SOC_avg*100))*(ST**0.8)/100
    return R_cell

def eff_volt(U_batt_t,R_cell,eff_sys,P,SOC,CE_batt,U_batt_nom,CE_cell,U_cell_nom):
    '''
    Calculate the voltage efficiency of the battery.

    Parameters
    ----------
    U_batt_t : float
        Open-circuit voltage of the battery for the dispatch interval.
    R_cell : float
        Internal resistance of the cells.
    eff_sys : float
        System efficiency accounting for auxiliary losses.
    P : float
        Dispatched power [MW].
    SOC : float
        State-of-charge of the system.
    CE_batt : float
        Energy capacity of the battery [MWh].
    U_batt_nom : float
        Nominal open-circuit-voltage of the battery.
    CE_cell : float
        Energy capacity of the cells [Wh].
    U_cell_nom : float
        Nominal open-circuit voltage of the cells.

    Returns
    -------
    eff_volt : float
        Voltage efficiency of the battery.

    '''
    
    # Calculate the radicand
    sqrt_var = 0.25 - (abs(P)/(eff_sys*SOC*CE_batt))*((U_batt_nom/U_batt_t)**2)*R_cell*CE_cell/U_cell_nom
    
    # Error handling for negative radicand
    if sqrt_var < 0:
        print(sqrt_var)
        eff_volt = 0.5
    
    # Voltage efficiency based on positive radicand
    else:
        eff_volt = 0.5 + (sqrt_var)**0.5
    
    return eff_volt

def SOC_max_aged(I_filt,I_cyc,SOC,SOC_previous,cell_capacity_Ah,Temp,delT,current_state): 
    '''
    Update the state of health in the current_state variable based on cycle or calendar ageing.
    Assumes all cells age at the same rate.

    Parameters
    ----------
    I_filt : float
        Current through the cells.
    I_cyc : float
        Current threshold for cycle ageing.
    SOC : float
        State of charge of the system in the current interval.
    SOC_previous : float
        State of charge of the system in the previous interval.
    cell_capacity_Ah : float
        Energy capacity of the cell [Ah].
    Temp : integer
        Temperature of the cells [K].
    delT : float
        Length of the dispatch interval [h].
    current_state : dictionary
        Dictionary of variables describing the current state of the system.

    Returns
    -------
    current_state : dictionary
        Dictionary of variables describing the current state of the system,
        including the updated state of health.

    '''
    # Define gas constant
    R = 8.314 #[J/mol/K]
    
    # Calculate cycle ageing based on empirical model and linear interpolation
    if I_filt < I_cyc:
        current_state["cycLossCurrentSum"] += I_filt
        current_state["cycLossIntervals"] += 1
        I_avg = abs(current_state["cycLossCurrentSum"]/current_state["cycLossIntervals"])
        current_state["Ah_throughput"] += (SOC - SOC_previous)*cell_capacity_Ah
        
        if I_avg <= 1:
            B_cyc = 3.16*10**3
            
        elif I_avg <= 4:
            B_cyc = 3.16*10**3 + (I_avg - 1)*(2.17*10**4-3.16*10**3)/(3) #Ah^(1-z_cyc)
            
        elif I_avg <= 12:
            B_cyc = 2.17*10**4 + (I_avg - 4)*(1.29*10**4-2.17*10**4)/(8)
        
        else:
            B_cyc = 1.29*10**4 + (I_avg - 12)*(1.55*10**4-1.29*10**4)/(8)
            
        z_cyc = 0.55
        Ea_cyc = 31700 #J/mol
        alpha = 370.3 #J/mol/A
        
        C_cyc_loss = B_cyc*np.exp((-Ea_cyc + alpha*I_avg)/(R*Temp))*((current_state["Ah_throughput"])**z_cyc) # % per cell
        current_state["SOC_max_loss_cyc"] = C_cyc_loss/100
    
    # Calculate calendar ageing based on empirical model and linear interpolation
    else:
        current_state["calLossTime"] += delT
        current_state["calLossIntervals"] += 1
        current_state["calLossCurrentSum"] += SOC
        SOC_avg = current_state["calLossCurrentSum"] / current_state["calLossIntervals"]
        if SOC_avg <= 0.3:
            B_cal = 7.34*10**5 # Ah/s^z_cal
            z_cal = 0.943 
            Ea_cal = 73369 #J/mol
            
        elif SOC_avg <= 0.65:
            B_cal = 7.34*10**5 + (SOC_avg-0.3)*(6.75*10**5-7.34*10**5)/(0.35) # Ah/s^z_cal
            z_cal = 0.943 + (SOC_avg-0.3)*(0.9-0.943)/0.35 
            Ea_cal = 73369 + (SOC_avg-0.3)*(69804-73369)/(0.35) #J/mol
            
        else:
            B_cal = 6.75*10**5 + (SOC_avg-0.65)*(2.18*10**5 - 6.75*10**5)/(0.35) # Ah/s^z_cal
            z_cal = 0.9 + (SOC_avg-0.65)*(0.683 - 0.9)/(0.35)
            Ea_cal = 69804 + (SOC_avg-0.65)*(56937 - 69804)/(0.35) #J/mol
        
        C_cal_loss = B_cal*np.exp(-Ea_cal/(R*Temp))*((current_state["calLossTime"]*3600)**z_cal) # % per cell
        current_state["SOC_max_loss_cal"] = C_cal_loss/100
    
    # Update the state of health of the system
    current_state["SOC_max"] = 1 - current_state["SOC_max_loss_cal"] - current_state["SOC_max_loss_cyc"]
        
    return current_state 

def pumpHeadLoss(system_assumptions,Q_pump_total):
    '''
    Calculate the pump head loss [m] for the PHS.

    Parameters
    ----------
    system_assumptions : dictionary
        Dictionary of the assumed system parameters for the simulation.
    Q_pump_total : float
        Flow rate in the pump penstock.

    Returns
    -------
    H_pl : float
        Pump head loss [m].

    '''
    
    # Initialize the parameters
    g = 9.81 # m/s^2
    rho = 997 # kg/m^3
    mu = 8.9*10**(-4) # Pa*s
    d_p = float(system_assumptions["pump_penstock_pipe_diameter"])
    L_p = float(system_assumptions["pump_penstock_pipe_length"])
    K_fittings = float(system_assumptions["pump_K_fittings"])
    abs_roughness = float(system_assumptions["pump_abs_roughness"])*10**(-3)
    
    # Define variables and calculate head loss
    v_p = Q_pump_total/(0.25*np.pi*(d_p**2))
    Re_p = (rho*v_p*d_p)/mu
    F_p = (1.8*np.log10(6.9/Re_p + ((abs_roughness/d_p)/3.7)**1.11))**(-2)
    K_pipe = (F_p*L_p)/d_p
    K_p = K_pipe + K_fittings
    H_pl = K_p*(v_p**2)/(2*g)
    
    return H_pl
      
def turbineHeadLoss(system_assumptions,Q_turbine_total):
    '''
    Calculate the turbine head loss [m] for the PHS.

    Parameters
    ----------
    system_assumptions : dictionary
        Dictionary of the assumed system parameters for the simulation.
    Q_turbine_total : float
        Flow rate in the turbine penstock.

    Returns
    -------
    H_tl : float
        Turbine head loss [m].

    '''
    
    # Initialize the parameters
    g = 9.81 # m/s^2
    rho = 997 # kg/m^3
    mu = 8.9*10**(-4) # Pa*s
    d_t = float(system_assumptions["turbine_penstock_pipe_diameter"])
    L_t = float(system_assumptions["turbine_penstock_pipe_length"])
    K_fittings = float(system_assumptions["turbine_K_fittings"])
    abs_roughness = float(system_assumptions["turbine_abs_roughness"])*10**(-3)
    
    # Define variables and calculate head loss
    if Q_turbine_total != 0:
        v_t = Q_turbine_total/(0.25*np.pi*(d_t**2))
        Re_t = (rho*v_t*d_t)/mu
        F_t = (1.8*np.log10(6.9/Re_t + ((abs_roughness/d_t)/3.7)**1.11))**(-2)
        K_pipe = (F_t*L_t)/d_t
        K_t = K_pipe + K_fittings
        H_tl = K_t*(v_t**2)/(2*g)
           
    else:
        H_tl = 0
    
    return H_tl

def pumpHead(SOC,system_assumptions,H_pl):
    '''
    Calculate the net head for the pumps.

    Parameters
    ----------
    SOC : float
        State-of-charge of the system.
    system_assumptions : dictionary
        Dictionary of the assumed system parameters for the simulation.
    H_pl : float
        Pump head loss [m].

    Returns
    -------
    pumpHead : float
        Net head of the pumps [m].

    '''
    
    # Initialize the parameters
    H_r = float(system_assumptions["H_r"])
    H_ur = float(system_assumptions["H_ur"])
    H_lr = float(system_assumptions["H_lr"])
    volume_ur = int(system_assumptions["V_res_upper"])
    volume_lr = int(system_assumptions["V_res_lower"])
    
    FSL_ur = H_r + H_ur + H_lr
    MOL_ur = float(system_assumptions["MOL_ur"])
    FSL_lr = H_lr
    MOL_lr = float(system_assumptions["MOL_lr"])
    
    # Calculate pump head
    pumpHead = (SOC*(FSL_ur - MOL_ur) + MOL_ur - ((1 - SOC)*(volume_ur/volume_lr)*(FSL_lr - MOL_lr)) - MOL_lr) + H_pl
    return pumpHead

def Q_pump(chargingPower,eff_pump,H_pump,rho,gravity):
    '''
    Flow rate for a pump unit.

    Parameters
    ----------
    chargingPower : float
        Dispatch power of the pump.
    eff_pump : float
        Pump efficiency.
    H_pump : float
        Net head of pump.
    rho : integer
        Density of water [kg/m^3].
    gravity : float
        Acceleration due to gravity [m/s^2].

    Returns
    -------
    Q_pump_t : float
        Pump flow rate during interval.

    '''
    
    if H_pump < 0:
        Q_pump_t = 0
        
    else:    
        Q_pump_t = (chargingPower*eff_pump*10**6)/(H_pump*rho*gravity)
    return Q_pump_t

def Q_turbine(dischargingPower,eff_turbine,H_turbine,rho,gravity):
    '''
    Flow rate for a turbine unit

    Parameters
    ----------
    dischargingPower : float
        Dispatch power of the turbine.
    eff_turbine : float
        Turbine efficiency.
    H_turbine : float
        Net head of turbine.
    rho : integer
        Density of water [kg/m^3].
    gravity : float
        Acceleration due to gravity [m/s^2].

    Returns
    -------
    Q_turbine_t : float
        Turbine flow rate during interval.

    '''
    
    if (eff_turbine == 0) or (H_turbine < 0):
        Q_turbine_t = 0
    
    else:
        Q_turbine_t = (dischargingPower*10**6)/(H_turbine*rho*gravity*eff_turbine)
        
    return Q_turbine_t

def turbineHead(SOC,system_assumptions,H_tl):
    '''
    Calculate net head for the turbines.

    Parameters
    ----------
    SOC : float
        State of charge of the system.
    system_assumptions : dictionary
        Dictionary of the assumed system parameters for the simulation.
    H_tl : float
        Turbine head loss [m].

    Returns
    -------
    turbineHead : float
        Net head of the turbines [m].

    '''
    # Define parameters
    H_r = float(system_assumptions["H_r"])
    H_ur = float(system_assumptions["H_ur"])
    H_lr = float(system_assumptions["H_lr"])
    volume_ur = int(system_assumptions["V_res_upper"])
    volume_lr = int(system_assumptions["V_res_lower"])
    
    FSL_ur = H_r + H_ur + H_lr
    MOL_ur = float(system_assumptions["MOL_ur"])
    FSL_lr = H_lr
    MOL_lr = float(system_assumptions["MOL_lr"])
    
    # Calculate turbine head
    turbineHead = (SOC*(FSL_ur - MOL_ur) + MOL_ur - ((1 - SOC)*(volume_ur/volume_lr)*(FSL_lr - MOL_lr)) - MOL_lr) - H_tl
    return turbineHead

def pumpEfficiency(Q_pump_g):
    '''
    Calculate the pump efficiency.

    Parameters
    ----------
    Q_pump_g : float
        Pump flow rate [m^3/s].

    Returns
    -------
    pump_efficiency : float
        Efficiency of the pump unit.

    '''
    pump_efficiency = 0.91
    return pump_efficiency

def turbineEfficiency(phs_assumptions,Q_turb_h,turbine_index):
    '''
    Calculate the turbine efficiency based on empirically fitted model.

    Parameters
    ----------
    phs_assumptions : dictionary
        Dictionary of the PHS pump and turbine assumptions.
    Q_turb_h : float
        Flow rate of turbine unit [m^3/s].
    turbine_index : integer
        Index 'h' of the turbine.

    Returns
    -------
    turbine_efficiency : float
        Efficiency of the turbine unit.

    '''
    # Define parameters
    Qpeak_turb_h = phs_assumptions["h"+str(turbine_index)]["Q_peak [m3/s]"]
    Q_ratio_turb_h = Q_turb_h / Qpeak_turb_h
    
    a = [-0.1421,3.044,-3.6718,2.3929,-0.7062]
    
    # Calculate turbine efficiency
    if Q_turb_h > 0.01:
        turbine_efficiency = sum(a[n]*(Q_ratio_turb_h**n) for n in range(0,len(a)))
    else:
        turbine_efficiency = 0
    return turbine_efficiency

def schedulingModel(system_assumptions,linearisation_df,SP,day,offer_PB,bid_PB,current_state,phs_assumptions,forecasting_horizon):
    '''
    Uses Pyomo library to define the model, then cbc (COIN-OR branch-and-cut) solver to optimise solution.
    Assumes dispatch capacities can be real numbers.

    Parameters
    ----------
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.
    linearisation_df : DataFrame
        Dataframe of piece-wise linear function parameters.
    SP : list
        Forecast spot prices for the scheduling period.
    day : integer
        Count of trading day being scheduled in the year.
    offer_PB : list
        Ordered list of generator offer price bands for the day.
    bid_PB : list
        Ordered list of load bid price bands for the day.
    current_state : dictionary
        Dictionary containing variables that define the current state of the system.
    phs_assumptions : dictionary
        Dictionary of the PHS pump and turbine assumptions.
    forecasting_horizon : integer
        Number of trading intervals that scheduling module optimises the MILP.

    Returns
    -------
    list
        List of bid and offer capacities, bid and offer price bands, and scheduled behaviour.

    '''
    
    # Define system level parameters
    system_type = system_assumptions["system_type"]
    riskLevel = int(system_assumptions["risk_level"])
    
    # Create abstract model object
    model = pyo.AbstractModel()
        
    # Declare index parameters and ranges
    s = forecasting_horizon # 48
    model.s = pyo.RangeSet(1,s)
        
    # Define fixed parameters
    model.mlf_load = pyo.Param(initialize=float(system_assumptions["mlf_load"]))
    model.mlf_gen = pyo.Param(initialize=float(system_assumptions["mlf_gen"]))
    model.dlf_load = pyo.Param(initialize=float(system_assumptions["dlf_load"]))
    model.dlf_gen = pyo.Param(initialize=float(system_assumptions["dlf_gen"]))
    model.delT = pyo.Param(initialize=(6*int(system_assumptions["Dispatch_interval_time"]))/60)
    model.SOCmax = pyo.Param(initialize=float(current_state["SOC_max"]))
    model.Pmin = pyo.Param(initialize=float(system_assumptions["P_min"]))
    model.Pmax = pyo.Param(initialize=float(system_assumptions["P_max"]))
    model.Ce = pyo.Param(initialize=int(system_assumptions["energy_capacity"]))
    model.SOCinitial = pyo.Param(initialize=current_state["SOC"])
    model.VOMd = pyo.Param(initialize=float(system_assumptions["VOMd"]))
    model.VOMc = pyo.Param(initialize=float(system_assumptions["VOMc"]))
    
    # Define system specific parameters/variables
    if system_type == 'General':
        # Declare binary variables
        model.ud = pyo.Var(model.s,within=pyo.Binary)
        model.uc = pyo.Var(model.s,within=pyo.Binary)
        
        # Declare decision variables
        model.D = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.C = pyo.Var(model.s,within=pyo.NonNegativeReals)
        
        # Declare parameters
        model.muCH = pyo.Param(initialize=float(system_assumptions["efficiency_ch_general"]))
        model.muDIS = pyo.Param(initialize=float(system_assumptions["efficiency_dis_general"]))
        model.SOCmin = pyo.Param(initialize=float(system_assumptions["SOC_min"]))
        
    elif system_type == 'BESS':
        # Declare index parameters and ranges
        j = 9
        i = 9
        m = 8
        model.j = pyo.RangeSet(1,j) # 9
        model.i = pyo.RangeSet(1,i) # 9
        model.m = pyo.RangeSet(1,m) # 8
        
        # Declare other parameters
        model.SOCmin = pyo.Param(initialize=float(system_assumptions["SOC_min"]))
        model.degC = pyo.Param(initialize=float(system_assumptions["DegC"]))
        
        # Declare binary variables
        model.w = pyo.Var(model.s,within=pyo.Binary)
        model.zc = pyo.Var(model.s,model.j,within=pyo.Binary)
        model.zd = pyo.Var(model.s,model.i,within=pyo.Binary)
        model.zcLoss = pyo.Var(model.s,model.m,within=pyo.Binary)
        model.zdLoss = pyo.Var(model.s,model.m,within=pyo.Binary)
        model.k = pyo.Var(model.s,within=pyo.Binary)
        
        # Declare linearisation parameters
        a_df = linearParameterDF(linearisation_df, system_type, 'a')
        b_df = linearParameterDF(linearisation_df, system_type, 'b')
        c_df = linearParameterDF(linearisation_df, system_type, 'c')
        d_df = linearParameterDF(linearisation_df, system_type, 'd')

        # Define SOC list for linearisation based on a and b
        SOC_list = np.cumsum(b_df)
        
        # Define power list for linearisation based on c and d
        Power_list = np.cumsum(c_df)
        
        # Define Pmax at each SOC        
        Pmax_list = []
        CE_batt = int(system_assumptions["energy_capacity"])
        Ucell_nom = U_OCV(0.5)
        eff_sys = float(system_assumptions["efficiency_sys"])
        Ubatt_nom = int(system_assumptions["series_cells"])*Ucell_nom
        R_cell = current_state["R_cell"]
        CE_cell = float(system_assumptions["cell_energy_capacity [Ah]"])*Ucell_nom
        
        for ii in range(0,len(a_df)):
            Ubatt_SOC = int(system_assumptions["series_cells"])*U_OCV(SOC_list[ii])
            Pmax_list.append(Pmax_ij_aged(SOC_list[ii],CE_batt,Ucell_nom,eff_sys,Ubatt_nom,Ubatt_SOC,R_cell,CE_cell) / int(system_assumptions["P_max"])) 
        
        # Define Ploss at each power
        P_loss_list = []
        for mm in range(0,len(c_df)):
            P_loss_list.append(Ploss_m_aged(Power_list[mm],CE_batt,Ucell_nom,eff_sys,R_cell,CE_cell)) 
        
        # Define all linearisation parameters
        def initialize_a(model,i):
            return {i:a_df[i-1] for i in model.i}
        model.a = pyo.Param(model.i,initialize=initialize_a,within=pyo.Any)
        
        def initialize_b(model,j):
            return {j:b_df[j-1] for j in model.j}
        model.b = pyo.Param(model.j,initialize=initialize_b,within=pyo.Any)
        
        def initialize_sc(model,j):
            return {j:(Pmax_list[j-1] - Pmax_list[j-2]) / b_df[j-1] if j > 1 else (Pmax_list[j-1]) / b_df[j-1] for j in model.j}
        model.sc = pyo.Param(model.j,initialize=initialize_sc,within=pyo.Any)
        
        def initialize_sd(model,i):
            return {i:(Pmax_list[i-1] - Pmax_list[i-2]) / a_df[i-1] if i > 1 else (Pmax_list[i-1]) / a_df[i-1] for i in model.i}
        model.sd = pyo.Param(model.i,initialize=initialize_sd,within=pyo.Any)
        
        def initialize_c(model,m):
            return {m:c_df[m-1] for m in model.m}
        model.c = pyo.Param(model.m,initialize=initialize_c,within=pyo.Any)
        
        def initialize_d(model,m):
            return {m:d_df[m-1] for m in model.m}
        model.d = pyo.Param(model.m,initialize=initialize_d,within=pyo.Any)
        
        def initialize_scLoss(model,m):
            return {m:(P_loss_list[m-1] - P_loss_list[m-2]) / d_df[m-1] if m > 1 else (P_loss_list[m-1]) / d_df[m-1] for m in model.m}
        model.scLoss = pyo.Param(model.m,initialize=initialize_scLoss,within=pyo.Any)
        
        def initialize_sdLoss(model,m):
            return {m:(P_loss_list[m-1] - P_loss_list[m-2]) / c_df[m-1] if m > 1 else (P_loss_list[m-1]) / c_df[m-1] for m in model.m}
        model.sdLoss = pyo.Param(model.m,initialize=initialize_sdLoss,within=pyo.Any)
        
        # Decalare other variables
        model.beta = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.x = pyo.Var(model.s,model.i,within=pyo.NonNegativeReals)
        model.y = pyo.Var(model.s,model.j,within=pyo.NonNegativeReals)
        model.alphac = pyo.Var(model.s,within=pyo.NonNegativeReals)   
        model.alphad = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.vc = pyo.Var(model.s,model.m,within=pyo.NonNegativeReals)
        model.vd = pyo.Var(model.s,model.m,within=pyo.NonNegativeReals)
        model.PcLoss = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.PdLoss = pyo.Var(model.s,within=pyo.NonNegativeReals)   
        model.Ed = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.Ec = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.D = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.C = pyo.Var(model.s,within=pyo.NonNegativeReals)
        
    elif system_type == 'PHS':
        # Declare index parameters and ranges
        j = 9
        i = 9
        m = 8
        g = int(system_assumptions["g_index_range"])
        h = int(system_assumptions["h_index_range"])
        model.m = pyo.RangeSet(1,m) # 8
        model.g = pyo.RangeSet(1,g) # 2
        model.h = pyo.RangeSet(1,h) # 2
        
        # Declare binary variables
        model.w = pyo.Var(model.s,within=pyo.Binary)
        model.zdLoss = pyo.Var(model.s,model.m,model.h,within=pyo.Binary)
        model.zcLoss = pyo.Var(model.s,model.m,model.g,within=pyo.Binary)
        
        # Declare linearisation parameters
        c_df = linearParameterDF(linearisation_df, system_type, 'c')
        d_df = linearParameterDF(linearisation_df, system_type, 'd')
        sdLoss_df = linearParameterDF(linearisation_df, system_type, 'sdLoss')
        scLoss_df = linearParameterDF(linearisation_df, system_type, 'scLoss')
        
        def initialize_c(model,m):
            return {m:c_df[m-1] for m in model.m}
        model.c = pyo.Param(model.m,initialize=initialize_c,within=pyo.Any)
        
        def initialize_d(model,m):
            return {m:d_df[m-1] for m in model.m}
        model.d = pyo.Param(model.m,initialize=initialize_d,within=pyo.Any)
        
        def initialize_sdLoss(model,m):
            return {m:sdLoss_df[m-1] for m in model.m}
        model.sdLoss = pyo.Param(model.m,initialize=initialize_sdLoss,within=pyo.Any)
        
        def initialize_scLoss(model,m):
            return {m:scLoss_df[m-1] for m in model.m}
        model.scLoss = pyo.Param(model.m,initialize=initialize_scLoss,within=pyo.Any)
                
        # Declare individual pump/turbine parameters      
        def initialize_QpPeak(model,g):
            return {g:phs_assumptions["g"+str(g)]["Q_peak [m3/s]"] for g in model.g}
        model.QpPeak = pyo.Param(model.g,initialize=initialize_QpPeak,within=pyo.Any)
        
        def initialize_QtPeak(model,h):
            return {h:phs_assumptions["h"+str(h)]["Q_peak [m3/s]"] for h in model.h}
        model.QtPeak = pyo.Param(model.h,initialize=initialize_QtPeak,within=pyo.Any)
        
        # Decalare other variables
        model.vd = pyo.Var(model.s,model.m,model.h,within=pyo.NonNegativeReals)
        model.vc = pyo.Var(model.s,model.m,model.g,within=pyo.NonNegativeReals)
        model.QtLoss = pyo.Var(model.s,model.h,within=pyo.NonNegativeReals) 
        model.QpLoss = pyo.Var(model.s,model.g,within=pyo.NonNegativeReals)
        model.Ed = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.Ec = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.Qp = pyo.Var(model.s,model.g,within=pyo.NonNegativeReals)
        model.Qt = pyo.Var(model.s,model.h,within=pyo.NonNegativeReals)
        
        # Declare model-specific parameters
        model.volume_reservoir = pyo.Param(initialize=int(system_assumptions["V_res_upper"]))
        model.rho = pyo.Param(initialize=997) # kg/m^3
        model.gravity = pyo.Param(initialize=9.8) #m/s^2
        model.Hpeffective = pyo.Param(initialize=float(system_assumptions["H_p_effective"]))
        model.Hteffective = pyo.Param(initialize=float(system_assumptions["H_t_effective"]))
        model.SOCmin = pyo.Param(initialize=float(system_assumptions["SOC_min"]))
        
        # Declare decision variables
        model.D = pyo.Var(model.s,model.h,within=pyo.NonNegativeReals)
        model.C = pyo.Var(model.s,model.g,within=pyo.NonNegativeReals)
        model.D_tot = pyo.Var(model.s,within=pyo.NonNegativeReals)
        model.C_tot = pyo.Var(model.s,within=pyo.NonNegativeReals)
    
    # Initialize the SP parameter
    def initializeSP(model,s):
        return {s:float(SP[s-1]) for s in model.s}
    model.SP = pyo.Param(model.s, initialize=initializeSP,within=pyo.Any)
    
    # Declare other decision variables
    model.SOC = pyo.Var(model.s)
    
    if system_type == 'BESS':
        # Constraint: Maximum charging power
        def CMax_rule(model,s):
            return model.C[s] <= -model.Pmin*model.w[s]
        model.CMax = pyo.Constraint(model.s,rule=CMax_rule)
        
        # Constraint: Maximum discharging power
        def DMax_rule(model,s):
            return model.D[s] <= model.Pmax*(1-model.w[s])
        model.DMax = pyo.Constraint(model.s,rule=DMax_rule)
        
        # Constraint: Subsequent SOC
        def subsequentSOC_rule(model,s):
            if s == 1:
                return model.SOC[s] == model.SOCinitial - (1/model.Ce)*(model.Ed[s] - model.Ec[s])    
            else:
                return model.SOC[s] == model.SOC[s-1] - (1/model.Ce)*(model.Ed[s] - model.Ec[s])
        model.subsequentSOC = pyo.Constraint(model.s, rule = subsequentSOC_rule)
        
        # Constraint: Maximum SOC
        def SOCMax_rule(model,s):
            return model.SOC[s] <= model.SOCmax
        model.SOCMax = pyo.Constraint(model.s,rule=SOCMax_rule)
        
        # Constraint: Maximum SOC
        def SOCMin_rule(model,s):
            return model.SOC[s] >= model.SOCmin
        model.SOCMin = pyo.Constraint(model.s,rule=SOCMin_rule)
        
        # Constraint: Define fraction of battery charged in trading interval
        def chargedFraction_rule(model,s):
            if s == 1:
                return model.beta[s] == (model.SOCinitial + model.SOC[s])/2
            else:
                return model.beta[s] == (model.SOC[s-1] + model.SOC[s])/2
        model.chargedFraction = pyo.Constraint(model.s,rule = chargedFraction_rule)
        
        # Constraint: Usable charging fraction
        def usableChargeFraction_rule(model,s):
            return model.alphac[s] == sum(model.y[s,j]*model.sc[j][j] for j in model.j)
        model.usableChargeFraction = pyo.Constraint(model.s,rule = usableChargeFraction_rule)
        
        # Constraint: Define value of beta
        def betaEqualityy_rule(model,s):
            return model.beta[s] == sum(model.y[s,j] for j in model.j)
        model.betaEqualityy = pyo.Constraint(model.s,rule = betaEqualityy_rule)
        
        # Constraint: Define maximum y
        def yMax_rule(model,s,j):
            return model.y[s,j] <= model.b[j][j]*model.zc[s,j]
        model.yMax = pyo.Constraint(model.s,model.j,rule = yMax_rule)
        
        # Constraint: Define minimum y
        def yMin_rule(model,s,j):
            if j >= 2:
                return model.y[s,j-1] >= model.b[j-1][j-1]*model.zc[s,j]
            else:
                return pyo.Constraint.Skip
        model.yMin = pyo.Constraint(model.s,model.j,rule = yMin_rule)
        
        # Constraint: Usable discharging fraction
        def usableDischargeFraction_rule(model,s):
            return model.alphad[s] == sum(model.x[s,i]*model.sd[i][i] for i in model.i)
        model.usableDischargeFraction = pyo.Constraint(model.s,rule = usableDischargeFraction_rule)
        
        # Constraint: Define value of beta
        def betaEqualityx_rule(model,s):
            return model.beta[s] == sum(model.x[s,i] for i in model.i)
        model.betaEqualityx = pyo.Constraint(model.s,rule = betaEqualityx_rule)
        
        # Constraint: Define maximum x
        def xMax_rule(model,s,i):
            return model.x[s,i] <= model.a[i][i]*model.zd[s,i]
        model.xMax = pyo.Constraint(model.s,model.i,rule = xMax_rule)
        
        # Constraint: Define minimum y
        def xMin_rule(model,s,i):
            if i >= 2:
                return model.x[s,i-1] >= model.a[i-1][i-1]*model.zd[s,i]
            else:
                return pyo.Constraint.Skip
        model.xMin = pyo.Constraint(model.s,model.i,rule = xMin_rule)
        
        # Constraint: define charging limit
        def chargingLimit_rule(model,s):
            return model.C[s] <= -model.Pmin*model.alphac[s]
        model.chargingLimit = pyo.Constraint(model.s,rule = chargingLimit_rule)
        
        # Constraint: define discharging limit
        def dischargingLimit_rule(model,s):
            return model.D[s] <= model.Pmax*model.alphad[s]
        model.dischargingLimit = pyo.Constraint(model.s,rule = dischargingLimit_rule)
        
        # Constraint: define discharged energy
        def chargedEnergy_rule(model,s):
            return model.Ec[s] == (model.C[s]-model.PcLoss[s])*model.delT
        model.chargedEnergy = pyo.Constraint(model.s,rule = chargedEnergy_rule)
        
        # Constraint: define charged energy
        def dischargedEnergy_rule(model,s):
            return model.Ed[s] == (model.D[s]+model.PdLoss[s])*model.delT
        model.dischargedEnergy = pyo.Constraint(model.s,rule = dischargedEnergy_rule)
        
        # Constraint: Define piecewise vc sum
        def vcSum_rule(model,s):
            return model.C[s] == sum(model.vc[s,m] for m in model.m)
        model.vcSum = pyo.Constraint(model.s,rule=vcSum_rule)
       
        # Constraint: Define vc pieces
        def vcPiece_rule(model,s,m):
           return model.vc[s,m] <= model.d[m][m]*model.zcLoss[s,m]
        model.vcPiece = pyo.Constraint(model.s,model.m,rule=vcPiece_rule)
        
        # Constraint: define subsequent vc pieces
        def vcPieceSubsequent_rule(model,s,m):
            if m >= 2:
                return model.vc[s,m-1] >= model.d[m-1][m-1]*model.zcLoss[s,m]
            else:
                return pyo.Constraint.Skip
        model.vcPieceSubsequent = pyo.Constraint(model.s,model.m,rule=vcPieceSubsequent_rule)
        
        # Constraint: Define piecewise vd sum
        def vdSum_rule(model,s):
            return model.D[s] == sum(model.vd[s,m] for m in model.m)
        model.vdSum = pyo.Constraint(model.s,rule=vdSum_rule)
       
        # Constraint: Define vd pieces
        def vdPiece_rule(model,s,m):
           return model.vd[s,m] <= model.c[m][m]*model.zdLoss[s,m]
        model.vdPiece = pyo.Constraint(model.s,model.m,rule=vdPiece_rule)
        
        # Constraint: define subsequent vd pieces
        def vdPieceSubsequent_rule(model,s,m):
            if m >= 2:
                return model.vd[s,m-1] >= model.c[m-1][m-1]*model.zdLoss[s,m]
            else:
                return pyo.Constraint.Skip
        model.vdPieceSubsequent = pyo.Constraint(model.s,model.m,rule=vdPieceSubsequent_rule)
        
        # Constraint: Maximum charge loss
        def chargeLossMax_rule(model,s):
            return model.PcLoss[s] <= -model.Pmin*model.w[s]
        model.chargeLossMax = pyo.Constraint(model.s,rule=chargeLossMax_rule)
        
        # Constraint: Maximum charge loss
        def dischargeLossMax_rule(model,s):
            return model.PdLoss[s] <= model.Pmax*(1-model.w[s])
        model.dischargeLossMax = pyo.Constraint(model.s,rule=dischargeLossMax_rule)
        
        # M2 Constraint: Charging loss value
        def chargeLossValue_rule(model,s):
            return model.PcLoss[s] == sum(model.vc[s,m]*model.scLoss[m][m] for m in model.m)
        model.chargeLossValue = pyo.Constraint(model.s,rule=chargeLossValue_rule)
        
        # M2 Constraint: Discharging loss value
        def dischargeLossValue_rule(model,s):
            return model.PdLoss[s] == sum(model.vd[s,m]*model.sdLoss[m][m] for m in model.m)
        model.dischargeLossValue = pyo.Constraint(model.s,rule=dischargeLossValue_rule)
        
        # Constraint: Final SOC
        def finalSOC_rule(model,s):
            if s == max(model.s):
                return model.SOC[s] == float(system_assumptions["SOC_initial"])
            else:
                return pyo.Constraint.Skip
        model.finalSOC = pyo.Constraint(model.s, rule=finalSOC_rule)
        
        # Objective: Maximise the arbitrage value for the day
        def arbitrageValue_rule(model):
            return sum(model.delT*model.SP[s][s]*(model.dlf_gen*model.mlf_gen*model.D[s] - model.dlf_load*model.mlf_load*model.C[s]) - (model.VOMd*model.Ed[s]) - (model.VOMc*model.Ec[s]) for s in model.s)
        model.arbitrageValue = pyo.Objective(rule=arbitrageValue_rule, sense = pyo.maximize)  
    
    elif system_type == 'PHS':
        
        # Constraint: Ensure pump only charges at peak flow rate
        def chargeMax_rule(model,s,g):
            return model.Qp[s,g] <= 1.05*model.QpPeak[g][g]*model.w[s]
        model.chargeMax = pyo.Constraint(model.s,model.g,rule=chargeMax_rule)
        
        # Constraint: Ensure pump only charges at peak flow rate
        def chargeMin_rule(model,s,g):
            return model.Qp[s,g] >= 0
        model.chargeMin = pyo.Constraint(model.s,model.g,rule=chargeMin_rule)
        
        # Constraint: Ensure discharging occurs within the flow rate operating range
        def dischargeMax_rule(model,s,h):
            return model.Qt[s,h] <= 1.05*model.QtPeak[h][h]*(1-model.w[s])
        model.dischargeMax = pyo.Constraint(model.s,model.h,rule=dischargeMax_rule)
        
        def dischargeMin_rule(model,s,h):
            return model.Qt[s,h] >= 0
        model.dischargeMin = pyo.Constraint(model.s,model.h,rule=dischargeMin_rule)
        
        # Constraint: Subsequent SOC
        def subsequentSOC_rule(model,s):
            if s == 1:
                return model.SOC[s] == model.SOCinitial + ((model.delT*3600)/(model.volume_reservoir))*(sum(model.Qp[s,g] for g in model.g) - sum(model.Qt[s,h] for h in model.h))
            else:
                return model.SOC[s] == model.SOC[s-1] + ((model.delT*3600)/(model.volume_reservoir))*(sum(model.Qp[s,g] for g in model.g) - sum(model.Qt[s,h] for h in model.h))
        model.subsequentSOC = pyo.Constraint(model.s, rule = subsequentSOC_rule)
        
        # Constraint: Ensure SOC does not exceed maximum
        def SOCMax_rule(model,s):
            return model.SOC[s] <= model.SOCmax
        model.SOCMax = pyo.Constraint(model.s,rule=SOCMax_rule)
        
        # Constraint: Ensure SOC does not drop below minimum
        def SOCMin_rule(model,s):
            return model.SOC[s] >= model.SOCmin 
        model.SOCMin = pyo.Constraint(model.s,rule=SOCMin_rule)
        
        # Constraint: define charged energy
        def chargedEnergy_rule(model,s):
            return model.Ec[s] == sum((model.C[s,g]-model.QpLoss[s,g]*model.Hpeffective*model.rho*model.gravity/(10**6)) for g in model.g)*model.delT
        model.chargedEnergy = pyo.Constraint(model.s,rule = chargedEnergy_rule)
        
        # Constraint: define discharged energy
        def dischargedEnergy_rule(model,s):
            return model.Ed[s] == sum((model.D[s,h]+model.QtLoss[s,h]*model.Hteffective*model.rho*model.gravity/(10**6)) for h in model.h)*model.delT
        model.dischargedEnergy = pyo.Constraint(model.s,rule = dischargedEnergy_rule)
        
        # Constraint: Define piecewise vd sum
        def vdSum_rule(model,s,h):
            return model.Qt[s,h] / model.QtPeak[h][h] == sum(model.vd[s,m,h] for m in model.m)
        model.vdSum = pyo.Constraint(model.s,model.h,rule=vdSum_rule)
       
        # Constraint: Define vd pieces
        def vdPiece_rule(model,s,m,h):
           return model.vd[s,m,h] <= model.c[m][m]*model.zdLoss[s,m,h]
        model.vdPiece = pyo.Constraint(model.s,model.m,model.h,rule=vdPiece_rule)
        
        # Constraint: define subsequent vd pieces
        def vdPieceSubsequent_rule(model,s,m,h):
            if m >= 2:
                return model.vd[s,m-1,h] >= model.c[m-1][m-1]*model.zdLoss[s,m,h]
            else:
                return pyo.Constraint.Skip
        model.vdPieceSubsequent = pyo.Constraint(model.s,model.m,model.h,rule=vdPieceSubsequent_rule)
        
        # Constraint: Activate discharge loss
        def dischargeLossMax_rule(model,s,h):
            return model.QtLoss[s,h] <= 1.2*model.QtPeak[h][h]*(1-model.w[s])
        model.dischargeLossMax = pyo.Constraint(model.s,model.h,rule=dischargeLossMax_rule)
        
        # M2 Constraint: Discharging loss value
        def dischargeLossValue_rule(model,s,h):
            return model.QtLoss[s,h] == model.QtPeak[h][h]*sum(model.vd[s,m,h]*model.sdLoss[m][m] for m in model.m)
        model.dischargeLossValue = pyo.Constraint(model.s,model.h,rule=dischargeLossValue_rule)

        # Constraint: Define piecewise vc sum
        def vcSum_rule(model,s,g):
            return model.Qp[s,g] / model.QpPeak[g][g] == sum(model.vc[s,m,g] for m in model.m)
        model.vcSum = pyo.Constraint(model.s,model.g,rule=vcSum_rule)
       
        # Constraint: Define vc pieces
        def vcPiece_rule(model,s,m,g):
           return model.vc[s,m,g] <= model.d[m][m]*model.zcLoss[s,m,g]
        model.vcPiece = pyo.Constraint(model.s,model.m,model.g,rule=vcPiece_rule)
        
        # Constraint: define subsequent vc pieces
        def vcPieceSubsequent_rule(model,s,m,g):
            if m >= 2:
                return model.vc[s,m-1,g] >= model.d[m-1][m-1]*model.zcLoss[s,m,g]
            else:
                return pyo.Constraint.Skip
        model.vcPieceSubsequent = pyo.Constraint(model.s,model.m,model.g,rule=vcPieceSubsequent_rule)
        
        # Constraint: Activate charge loss
        def chargeLossMax_rule(model,s,g):
            return model.QpLoss[s,g] <= 1.2*model.QpPeak[g][g]*model.w[s]
        model.chargeLossMax = pyo.Constraint(model.s,model.g,rule=chargeLossMax_rule)
        
        # M2 Constraint: Charging loss value
        def chargeLossValue_rule(model,s,g):
            return model.QpLoss[s,g] == model.QpPeak[g][g]*sum(model.vc[s,m,g]*model.scLoss[m][m] for m in model.m)
        model.chargeLossValue = pyo.Constraint(model.s,model.g,rule=chargeLossValue_rule)
        
        # Constraint: Total charge power
        def totalChargePower_rule(model,s):
            return model.C_tot[s] == sum(model.C[s,g] for g in model.g)
        model.totalChargePower = pyo.Constraint(model.s,rule=totalChargePower_rule)
        
        # Constraint: Total discharge power
        def totalDischargePower_rule(model,s):
            return model.D_tot[s] == sum(model.D[s,h] for h in model.h)
        model.totalDischargePower = pyo.Constraint(model.s,rule=totalDischargePower_rule)
        
        # Constraint: Water flow rate pumping
        def pumpFlow_rule(model,s,g):
            return model.Qp[s,g] == ((model.C[s,g]*10**6)/(model.Hpeffective*model.rho*model.gravity))-model.QpLoss[s,g]
        model.pumpFlow = pyo.Constraint(model.s,model.g,rule=pumpFlow_rule)
        
        # Constraint: Water flow rate turbine
        def turbineFlow_rule(model,s,h):
            return model.Qt[s,h] == ((model.D[s,h]*10**6)/(model.Hteffective*model.rho*model.gravity))+model.QtLoss[s,h]
        model.turbineFlow = pyo.Constraint(model.s,model.h,rule=turbineFlow_rule)
        
        # Constraint: Final SOC
        def finalSOC_rule(model,s):
            if s == max(model.s):
                return model.SOC[s] == float(system_assumptions["SOC_initial"])
            else:
                return pyo.Constraint.Skip
        model.finalSOC = pyo.Constraint(model.s, rule=finalSOC_rule)
        
        # Objective: Maximise the arbitrage value for the day
        def arbitrageValue_rule(model):
            return sum(model.delT*model.SP[s][s]*(model.dlf_gen*model.mlf_gen*model.D_tot[s] - model.dlf_load*model.mlf_load*model.C_tot[s]) - (model.VOMd*model.Ed[s]) - (model.VOMc*model.Ec[s]) for s in model.s) 
        model.arbitrageValue = pyo.Objective(rule=arbitrageValue_rule, sense = pyo.maximize) 
        
    else:
        print('INCORRECTLY DEFINED STORAGE SYSTEM TYPE')

    # Use solver to output optimal decision variables. Results moved to instance by default after solve method used
    instance = model.create_instance()
    
    # solverpath_exe='path/to/cbc'
    solverpath_exe='C:\\Users\\peckh\\anaconda3\\Library\\bin\\cbc'
    
    opt = pyo.SolverFactory('cbc',executable=solverpath_exe)
    opt.options['seconds'] = 1200
    result = opt.solve(instance)
    result.Solver.Status = SolverStatus.warning
    print("scheduling successful")
    
    # Define the output variables
    dispatch_offer_cap = []
    dispatch_bid_cap = []
    Total_dispatch_cap = []
    dispatch_offers = []
    dispatch_bids = []
    unit_g_capacities = {}
    unit_h_capacities = {}
    EnergyD = []
    EnergyC = []
    ws = []
    
    if system_type == "PHS":
    
        for g in instance.g:
            unit_g_capacities[str(g)] = []
            
        for h in instance.h:
            unit_h_capacities[str(h)] = []
    
    for d in range(1,49):
        if system_type == "PHS":
            unit_h_subOffers = []
            unit_h_subFlows = []
            unit_h_losses = []
            unit_g_subBids = []
            # Optimisation outputs
            for h in instance.h:
                unit_h_subOffers.append(instance.D[d,h].value)  
                unit_h_subFlows.append(instance.Qt[d,h].value)
                unit_h_losses.append(instance.QtLoss[d,h].value)
                unit_h_capacities[str(h)].append(instance.D[d,h].value)
                
            for g in instance.g:
                if instance.C[d,g].value > 0.1:
                    unit_g_subBids.append(-int(phs_assumptions["g"+str(g)]["P_rated [MW]"]))
                    unit_g_capacities[str(g)].append(-int(phs_assumptions["g"+str(g)]["P_rated [MW]"]))
                else:
                    unit_g_subBids.append(0)
                    unit_g_capacities[str(g)].append(0)
                
            dispatch_offer_cap.append(unit_h_subOffers)
            dispatch_bid_cap.append(unit_g_subBids)
            
            # Define dispatch bid/offer for correlation 
            if instance.D_tot[d].value > 0:
                Total_dispatch_cap.append(instance.D_tot[d].value)
            else:
                Total_dispatch_cap.append(-instance.C_tot[d].value)
            
        else:
            dispatch_offer_cap.append(instance.D[d].value)
            dispatch_bid_cap.append(-instance.C[d].value)
            
            # Define dispatch bid/offer for correlation 
            if instance.D[d].value > 0:
                Total_dispatch_cap.append(instance.D[d].value)
            else:
                Total_dispatch_cap.append(-instance.C[d].value)
            
        dispatch_offers.append(max_lessThan(offer_PB,float(instance.SP[d][d]),riskLevel))
        dispatch_bids.append(min_greaterThan(bid_PB,float(instance.SP[d][d]),riskLevel))
        ws.append(instance.w[d].value)
        EnergyD.append(instance.Ed[d].value)
        EnergyC.append(instance.Ec[d].value)
        
    return [dispatch_offer_cap,dispatch_bid_cap,dispatch_offers,dispatch_bids,ws,Total_dispatch_cap]

def dispatchModel(dispatch_bidsOffers, dispatch_prices,spot_prices,system_assumptions):
    '''
    Generate the dispatch instructions for the trading day.

    Parameters
    ----------
    dispatch_bidsOffers : list
        Outputs from the scheduling model.
    dispatch_prices : list
        List of dispatch prices for the trading day.
    spot_prices : list
        List of historical spot prices for the trading day.
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.

    Returns
    -------
    list
        List containing dispatch instructions for the trading day.

    '''
    # Define parameters
    dispatch_offers = dispatch_bidsOffers[2]
    dispatch_bids = dispatch_bidsOffers[3]
    dispatch_offer_cap = dispatch_bidsOffers[0]
    dispatch_bid_cap = dispatch_bidsOffers[1]
    ws = dispatch_bidsOffers[4]
    Total_dispatch_cap = dispatch_bidsOffers[5]
    g_pumps = int(system_assumptions["g_index_range"])
    h_turbines = int(system_assumptions["h_index_range"])
    system_type = system_assumptions["system_type"]
    
    # Define output
    dispatchInstructions = []
    
    # Determine if bids/offers are won for each dispatch interval. Define dispatch instructions accordingly
    for s in range(0,48):
        for t in range(0,6):
            if ws[s] == 0:
                if dispatch_offers[s] <= dispatch_prices[t+6*s]:
                    dispatchInstructions.append(dispatch_offer_cap[s])
                else:
                    if system_type == "PHS":
                        dispatchInstructions.append(h_turbines*[0])
                    else:
                        dispatchInstructions.append(0)
                
            else:
                if dispatch_bids[s] >= dispatch_prices[t+6*s]:
                    dispatchInstructions.append(dispatch_bid_cap[s])
                else:
                    if system_type == "PHS":
                        dispatchInstructions.append(g_pumps*[0])
                    else:
                        dispatchInstructions.append(0)
                    
    return [dispatchInstructions]

def chargingModel(current_state,system_assumptions,dispatchInstructions,day,phs_assumptions,year):
    '''
    Perform the charging/discharging according to the dispatch instructions for the trading day.

    Parameters
    ----------
    current_state : dictionary
        Dictionary containing the variables that define the current state of the system.
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.
    dispatchInstructions : list
        List of outputs from the dispatch model.
    day : integer
        Count of days in the year on the current trading day.
    phs_assumptions : dictionary
        Dictionary of assumed parameters for the PHS turbines and pumps.
    year : integer
        Count of the year iterations on the current trading day.

    Returns
    -------
    list
        List of outputs at the end of the trading day.

    '''
    
    # Define system level parameters
    cycle_tracker = current_state["cycle_tracker"]
    daily_cycles = 0
    system_type = system_assumptions["system_type"]
    
    # Define charging parameters
    SOC_min = float(system_assumptions["SOC_min"])
    delT = int(system_assumptions["Dispatch_interval_time"])/60
    SOC_initial = current_state["SOC"]
    
    # Define starting charging and discharging powers
    if current_state["Power"] < 0:
        C_initial = -current_state["Power"]
        D_initial = 0
    else:
        D_initial = current_state["Power"]
        C_initial = 0
    
    # Define lists for tracking variables during the trading day
    SOC_day = [SOC_initial]
    chargingCapacity = [C_initial]
    dischargingCapacity = [D_initial]
    behaviour = [] # charge = -1, standby = 0, discharge = 1
    headLossPump = []
    headLossTurbine = []
    headPump = []
    headTurbine = []
    flowRatePump = []
    flowRateTurbine = []
    efficiencyTurbineDay = []
    dischargedEnergy = []
    chargedEnergy = []
    U_batt_day = []
    eff_volt_day = []
    calendarLossDay = []
    cycleLossDay = []
    R_cell_day = []
    SOC_max_day = []
    
    # Perform charging/discharging operation for each dispatch interval 
    for t in range(0,len(dispatchInstructions)):
        # Define the previous interval's SOC
        if t == 0:
            SOC_t_pre = SOC_initial
        else:
            SOC_t_pre = SOC_day[-1]
        
        # Simple charging model
        if system_type == 'General':
            
            # Define system parameters
            Ce = int(system_assumptions["energy_capacity_faded"])
            rho_ch = float(system_assumptions["efficiency_ch_general"])
            rho_dis = float(system_assumptions["efficiency_dis_general"])
                
            # Create the test SOC update
            if dispatchInstructions[t] < 0:
                SOC_exp = SOC_t_pre - (1/Ce)*rho_ch*dispatchInstructions[t]*delT
                behaviour.append(-1)
            elif dispatchInstructions[t] > 0:
                SOC_exp = SOC_t_pre - (1/Ce)*(1/rho_dis)*dispatchInstructions[t]*delT 
                behaviour.append(1)
            else:
                SOC_exp = SOC_t_pre
                behaviour.append(0)
    
        # Perform the battery charging operation
        elif system_type == 'BESS':
        
            # Define BESS parameters
            Ce = int(system_assumptions["energy_capacity"])
            eff_sys = float(system_assumptions["efficiency_sys"])
            Temp = int(system_assumptions["Temp"])
            R_cell_initial = float(system_assumptions["R_cell_initial"])
            I_cyc = 0
            N_series = int(system_assumptions["series_cells"])
            
            # Define interval variables
            U_cell = U_OCV(SOC_t_pre)
            U_cell_nom = U_OCV(0.5)
            CE_cell = float(system_assumptions["cell_energy_capacity [Ah]"])*U_cell_nom
            R_cell_t_pre = R_cell(year,day,t,Temp,R_cell_initial,current_state)
            U_batt = U_cell*N_series
            U_batt_day.append(U_batt)
            U_batt_nom = U_cell_nom*N_series
            eff_volt_t = eff_volt(U_batt,R_cell_t_pre,eff_sys,dispatchInstructions[t],SOC_t_pre,Ce,U_batt_nom,CE_cell,U_cell_nom)
            eff_volt_day.append(eff_volt_t)
            
            # Create the test SOC update
            if dispatchInstructions[t] < 0:
                SOC_exp = SOC_t_pre - (1/Ce)*eff_volt_t*eff_sys*dispatchInstructions[t]*delT
                behaviour.append(-1)
            elif dispatchInstructions[t] > 0:
                SOC_exp = SOC_t_pre - (1/Ce)*(1/(eff_volt_t*eff_sys))*dispatchInstructions[t]*delT
                behaviour.append(1)
            else:
                SOC_exp = SOC_t_pre
                behaviour.append(0)
        
        # Perform the pumped hydro charging operation
        elif system_type == 'PHS':
            
            # Define PHS parameters
            volume_reservoir = int(system_assumptions["V_res_upper"])
            rho = 997 # kg/m^3
            gravity = 9.81 # m/s^2
            g_index_range = int(system_assumptions["g_index_range"])
            h_index_range = int(system_assumptions["h_index_range"])
            rampTime_T_TNL = int(system_assumptions["RT_T_TNL"])
            rampTime_TNL_P = int(system_assumptions["RT_TNL_P"])
            rampTime_P_TNL = int(system_assumptions["RT_P_TNL"])
            rampTime_TNL_T = int(system_assumptions["RT_TNL_T"])
            Q_p_list_previous = current_state["Q_p_list_previous"]
            Q_t_list_previous = current_state["Q_t_list_previous"]
            
            # Initialise parameters
            H_pl_initial = 0
            H_tl_initial = 0
            head_turb = turbineHead(SOC_t_pre,system_assumptions,H_tl_initial)
            head_pump = pumpHead(SOC_t_pre,system_assumptions,H_pl_initial)
            efficiency_pump = 0.91
            efficiency_turbine = 0.91
            
            # Charging behaviour
            if (sum(dispatchInstructions[t]) < 0):
                Q_p_pre_newSum = Q_pump(0,efficiency_pump,head_pump,rho,gravity)
                Q_t_pre_newSum = 0
                H_pl = -1
                Q_t_list = h_index_range*[0]
                
                # Search for steady-state pump variable values
                while abs(H_pl - H_pl_initial) > 0.01*H_pl:
                    Q_p_list = []
                    H_pl_initial = H_pl
                    # Calculate pump flow rates
                    for g in range(0,g_index_range):   
                        Q_p_g = Q_pump(-dispatchInstructions[t][g],efficiency_pump,head_pump,rho,gravity)
                        efficiency_pump = pumpEfficiency(Q_p_g)
                        Q_p_list.append(Q_p_g)
                    # Send pump flows to pump penstock
                    Q_p_pre_newSum = sum(Q_p_list)
                    H_pl = pumpHeadLoss(system_assumptions,Q_p_pre_newSum)
                    head_pump = pumpHead(SOC_t_pre,system_assumptions,H_pl)
                
                # Calculate new SOC
                SOC_exp = ((delT*3600)/volume_reservoir)*(Q_p_pre_newSum - Q_t_pre_newSum)+SOC_t_pre
                
                # Append variables to lists
                behaviour.append(-1)
                headLossPump.append(H_pl)
                headPump.append(head_pump)
                flowRatePump.append(Q_p_pre_newSum)
                headLossTurbine.append(0)
                headTurbine.append(0)
                flowRateTurbine.append(0)
                efficiencyTurbineDay.append(0)
                       
            elif (sum(dispatchInstructions[t]) > 1):                
                Q_p_pre_newSum = 0
                Q_t_pre_newSum = Q_turbine(0,efficiency_turbine,head_pump,rho,gravity)
                H_tl = -1
                Q_p_list = g_index_range*[0]
                turb_eff_list = h_index_range*[efficiency_turbine]
                
                # Search for steady-state turbine variable values
                while abs(H_tl - H_tl_initial) > 0.01*H_tl:
                    Q_t_list = []
                    
                    H_tl_initial = H_tl
                    # Calculate turbine flow rates
                    for h in range(0,h_index_range):   
                        Q_t_h = Q_turbine(dispatchInstructions[t][h],turb_eff_list[h],head_turb,rho,gravity)
                        efficiency_turbine = turbineEfficiency(phs_assumptions,Q_t_h,h+1)
                        Q_t_list.append(Q_t_h)
                        turb_eff_list[h] = efficiency_turbine
                        
                    # Send turbine flows to turbine penstock  
                    Q_t_pre_newSum = sum(Q_t_list)
                    H_tl = turbineHeadLoss(system_assumptions,Q_t_pre_newSum)
                    head_turb = turbineHead(SOC_t_pre,system_assumptions,H_tl)
                    totalTurbEff = 0
                    
                    # Calculate the overall turbine efficiency
                    for a in range(0,len(Q_t_list)):
                        
                        effProp = (Q_t_list[a] / Q_t_pre_newSum) * turb_eff_list[a]
                        totalTurbEff += effProp
                
                # Update discharging variables for the dispatch interval
                SOC_exp = ((delT*3600)/volume_reservoir)*(Q_p_pre_newSum - Q_t_pre_newSum)+SOC_t_pre
                behaviour.append(1)
                headLossTurbine.append(H_tl)
                headTurbine.append(head_turb)
                flowRateTurbine.append(Q_t_pre_newSum)
                efficiencyTurbineDay.append(totalTurbEff)
                headLossPump.append(0)
                headPump.append(0)
                flowRatePump.append(0)
                
            else:
                # Update variables if system is idleing
                Q_p_list = g_index_range*[0]
                Q_t_list = h_index_range*[0]
                SOC_exp = SOC_t_pre
                behaviour.append(0) 
                headLossTurbine.append(0)
                headTurbine.append(0)
                flowRateTurbine.append(0)
                headLossPump.append(0)
                headPump.append(0)
                flowRatePump.append(0)
                efficiencyTurbineDay.append(0)
                
            # Calculate ramp times
            V_transient_adjust_t = []
            V_transient_adjust_p = []
            RT_t = []
            RT_p = []
                             
            if (sum(Q_t_list) == sum(Q_t_list_previous)):
            
                for g in range(0,g_index_range):
                    Q_p_peak = float(phs_assumptions["g"+str(g+1)]["Q_peak [m3/s]"])
                    if sum(Q_p_list) < sum(Q_p_list_previous):
                        RT_p_g = np.abs(Q_p_list[g] - Q_p_list_previous[g])/Q_p_peak * rampTime_P_TNL
                    else:
                        RT_p_g = np.abs(Q_p_list[g] - Q_p_list_previous[g])/Q_p_peak * rampTime_TNL_P
                    RT_p.append(RT_p_g)
                    V_transient_adjust_p_g = RT_p_g*(Q_p_list_previous[g] - Q_p_list[g])/2
                    V_transient_adjust_p.append(V_transient_adjust_p_g)
                
                V_transient_adjust_t = h_index_range*[0]
                RT_t = h_index_range*[0]
                
            elif (sum(Q_p_list) == sum(Q_p_list_previous)):
                
                V_transient_adjust_p = g_index_range*[0]
                RT_p = g_index_range*[0]
                
                for h in range(0,h_index_range):
                    Q_t_peak = float(phs_assumptions["h"+str(h+1)]["Q_peak [m3/s]"])
                    if sum(Q_t_list) < sum(Q_t_list_previous):
                        RT_t_h = np.abs(Q_t_list_previous[h] - Q_t_list[h])/Q_t_peak * rampTime_T_TNL
                    else: 
                        RT_t_h = np.abs(Q_t_list_previous[h] - Q_t_list[h])/Q_t_peak * rampTime_TNL_T
                    RT_t.append(RT_t_h)
                    V_transient_adjust_t_h = RT_t_h*(Q_t_list[h] - Q_t_list_previous[h])/2
                    V_transient_adjust_t.append(V_transient_adjust_t_h)
                    
            elif (sum(Q_p_list) < sum(Q_p_list_previous)) and (sum(Q_t_list) > sum(Q_t_list_previous)):
                
                for g in range(0,g_index_range):
                    Q_p_peak = float(phs_assumptions["g"+str(g+1)]["Q_peak [m3/s]"])
                    RT_p_g = np.abs(Q_p_list[g] - Q_p_list_previous[g])/Q_p_peak * rampTime_P_TNL
                    RT_p.append(RT_p_g)
                    V_transient_adjust_p_g = RT_p_g*(Q_p_list_previous[g] - Q_p_list[g])/2
                    V_transient_adjust_p.append(V_transient_adjust_p_g)
                    
                for h in range(0,h_index_range):
                    Q_t_peak = float(phs_assumptions["h"+str(h+1)]["Q_peak [m3/s]"])
                    RT_t_h = np.abs(Q_t_list_previous[h] - Q_t_list[h])/Q_t_peak * rampTime_TNL_T
                    RT_t.append(RT_t_h)
                    V_transient_adjust_t_h = (RT_t_h/2 + max(RT_p))*(Q_t_list[h] - Q_t_list_previous[h]) 
                    V_transient_adjust_t.append(V_transient_adjust_t_h)
                    
            elif (sum(Q_p_list) > sum(Q_p_list_previous)) and (sum(Q_t_list) < sum(Q_t_list_previous)):
                
                for h in range(0,h_index_range):
                    Q_t_peak = float(phs_assumptions["h"+str(h+1)]["Q_peak [m3/s]"])
                    RT_t_h = np.abs(Q_t_list_previous[h] - Q_t_list[h])/Q_t_peak * rampTime_T_TNL
                    RT_t.append(RT_t_h)
                    V_transient_adjust_t_h = RT_t_h*(Q_t_list[h] - Q_t_list_previous[h])/2 
                    V_transient_adjust_t.append(V_transient_adjust_t_h)
                
                for g in range(0,g_index_range):
                    Q_p_peak = float(phs_assumptions["g"+str(g+1)]["Q_peak [m3/s]"])
                    RT_p_g = np.abs(Q_p_list[g] - Q_p_list_previous[g])/Q_p_peak * rampTime_TNL_P
                    RT_p.append(RT_p_g)
                    V_transient_adjust_p_g = (RT_p_g/2 + max(RT_t))*(Q_p_list_previous[g] - Q_p_list[g])
                    V_transient_adjust_p.append(V_transient_adjust_p_g)
            
            else:
                V_transient_adjust_t = h_index_range*[0]
                V_transient_adjust_p = g_index_range*[0]
                RT_t = h_index_range*[0]
                RT_p = g_index_range*[0]
            
            # Update the SOC with the transient adjustments
            SOC_exp += (1/volume_reservoir)*(sum(V_transient_adjust_p) + sum(V_transient_adjust_t))
            
        # Determine the actual SOC update
        if system_type == "PHS":
            SOC_max = current_state["SOC_max"]
            
            if behaviour[t] == -1 and SOC_exp <= SOC_max:
                cycle_tracker = 0
                chargingCapacity.append(dispatchInstructions[t])
                dischargingCapacity.append([0]*h_index_range)
                chargedEnergy.append(-sum(dispatchInstructions[t])*delT + sum((RT_p[g]/2 + max(RT_t))/3600*(-current_state["P_p_list_previous"][g] + dispatchInstructions[t][g]) for g in range(0,g_index_range)))
                dischargedEnergy.append(sum((RT_t[h]/2)/3600*(current_state["P_t_list_previous"][h]) for h in range(0,h_index_range)))
                SOC_day.append(SOC_exp)
                final_capacity = sum(dispatchInstructions[t])
                current_state["Q_p_list_previous"] = Q_p_list
                current_state["Q_t_list_previous"] = Q_t_list
                current_state["P_p_list_previous"] = dispatchInstructions[t] 
                current_state["P_t_list_previous"] = [0]*h_index_range
            elif behaviour[t] == 1 and SOC_exp >= SOC_min:
                if cycle_tracker == 0:
                    daily_cycles += 1
                    cycle_tracker = 1
                chargingCapacity.append([0]*g_index_range)
                dischargingCapacity.append(dispatchInstructions[t])
                chargedEnergy.append(-sum((RT_p[g]/2)/3600*(current_state["P_p_list_previous"][g]) for g in range(0,g_index_range)))
                dischargedEnergy.append(sum(dispatchInstructions[t])*delT + sum((RT_t[h]/2 + max(RT_p))/3600*(current_state["P_t_list_previous"][h] - dispatchInstructions[t][h]) for h in range(0,h_index_range)))
                SOC_day.append(SOC_exp)
                final_capacity = sum(dispatchInstructions[t])
                current_state["Q_p_list_previous"] = Q_p_list
                current_state["Q_t_list_previous"] = Q_t_list
                current_state["P_p_list_previous"] = [0]*g_index_range
                current_state["P_t_list_previous"] = dispatchInstructions[t] 
            else:
                chargingCapacity.append([0]*g_index_range)
                dischargingCapacity.append([0]*h_index_range)
                chargedEnergy.append(0)
                dischargedEnergy.append(0)
                SOC_day.append(SOC_day[-1])
                final_capacity = 0
                current_state["Q_p_list_previous"] = [0]*g_index_range
                current_state["Q_t_list_previous"] = [0]*h_index_range
                current_state["P_p_list_previous"] = [0]*g_index_range
                current_state["P_t_list_previous"] = [0]*h_index_range
        
        else:
            SOC_max = current_state["SOC_max"]
            
            if behaviour[t] == -1 and SOC_exp <= SOC_max:
                cycle_tracker = 0
                chargingCapacity.append(dispatchInstructions[t])
                dischargingCapacity.append(0)
                chargedEnergy.append(-dispatchInstructions[t]*delT)
                dischargedEnergy.append(0)
                SOC_day.append(SOC_exp)
                final_capacity = dispatchInstructions[t]
            elif behaviour[t] == 1 and SOC_exp >= SOC_min:
                if cycle_tracker == 0:
                    daily_cycles += 1
                    cycle_tracker = 1
                chargingCapacity.append(0)
                dischargingCapacity.append(dispatchInstructions[t])                 
                chargedEnergy.append(0)
                dischargedEnergy.append(dispatchInstructions[t]*delT)
                SOC_day.append(SOC_exp)
                final_capacity = dispatchInstructions[t]
            else:
                chargingCapacity.append(0)
                dischargingCapacity.append(0)
                chargedEnergy.append(0)
                dischargedEnergy.append(0)
                SOC_day.append(SOC_day[-1])
                final_capacity = 0
        
        # Capacity fading
        if system_type == "BESS":
            SOC = SOC_day[-1]
            SOC_previous = SOC_day[-2]
            U_ocv = N_series*U_OCV(SOC)
            I_cell = 10**6*(chargingCapacity[-1]+dischargingCapacity[-1])/(74*6*8*33*U_ocv)
            current_state = SOC_max_aged(I_cell,I_cyc,SOC,SOC_previous, float(system_assumptions["cell_energy_capacity [Ah]"]), Temp,delT,current_state)
            current_state["SOC_sum"] += SOC
            current_state["dispatch_intervals"] += 1
            
            # Efficiency fading
            R_cell_t = R_cell(year,day,t,Temp,R_cell_initial,current_state)
            current_state["R_cell"] = R_cell_t
            
            calendarLossDay.append(current_state["SOC_max_loss_cal"])
            cycleLossDay.append(current_state["SOC_max_loss_cyc"])
            R_cell_day.append(current_state["R_cell"])
            SOC_max_day.append(current_state["SOC_max"])            
    
    # Remove the initial values from the list
    SOC_day.pop(0)
    chargingCapacity.pop(0)
    dischargingCapacity.pop(0)
    
    # Store the cycle_tracker
    current_state["cycle_tracker"] = cycle_tracker
    
    return [SOC_day,dischargingCapacity,chargingCapacity,final_capacity,dischargedEnergy,chargedEnergy,daily_cycles]

    
def settlementModel(system_assumptions,energy,SP):
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
    mlf_load = float(system_assumptions["mlf_load"])
    mlf_gen = float(system_assumptions["mlf_gen"])
    dlf_load = float(system_assumptions["dlf_load"])
    dlf_gen = float(system_assumptions["dlf_gen"])
    dischargedEnergy = energy[0]
    chargedEnergy = energy[1]
    TA_ch = []
    TA_dis = []
    
    for s in range(0,48):
        for t in range(0,6):
            
            # Calculate the trade revenue for the dispatch interval
            TA_s = (dlf_gen*mlf_gen*dischargedEnergy[t+6*s]-dlf_load*mlf_load*chargedEnergy[t+6*s])*SP[s]
            if TA_s < 0:
                TA_ch.append(TA_s)
                TA_dis.append(0)
            else:
                TA_ch.append(0)
                TA_dis.append(TA_s)
    
    TA = [TA_dis,TA_ch]
    
    return TA

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
    
def dailySimulation(system_assumptions,linearisation_df,SP,DP,day,year,current_state,total_days_cumulative,phs_assumptions,year_count,imperfectSP,forecasting_horizon,offer_list,bid_list):
    '''
    Simulate arbitrage over one trading day.

    Parameters
    ----------
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.
    linearisation_df : DataFrame
        Dataframe of the linearisation parameters for the piecewise linear functions.
    SP : list
        List of spot prices for all trading intervals in the year.
    DP : list
        List of dispatch prices for all trading intervals in the year.
    day : integer
        Count of days in the year at the current trading day.
    year : integer
        Year for which price data belongs.
    current_state : dictionary
        Dictionary of variables describing the current state of the system at the start of the trading day.
    total_days_cumulative : integer
        Count of days in the simulation at the current trading day.
    phs_assumptions : dictionary
        Dictionary of assumed parameters for the PHS turbines and pumps.
    year_count : integer
        Count of iterations of the year at the current trading day.
    imperfectSP : list
        List of imperfectly forecast spot prices for all trading intervals in the year.
    forecasting_horizon : integer
        Number of trading intervals which the scheduling model optimises at once.
    offer_list : list
        List of lists containing generator offer price bands for each day in the year.
    bid_list : list
        List of lists containing load price bands for each day in the year.

    Returns
    -------
    list
        Outputs at the end of the trading day.

    '''
    
    # Select the prices for the trading day
    SP_day = SP[(total_days_cumulative*48+day*48):(total_days_cumulative*48+48+day*48)]
    imperfectSP_day = imperfectSP[(total_days_cumulative*48+day*48):(total_days_cumulative*48+forecasting_horizon+day*48)]
    DP_day = DP[(day*288):(288+day*288)]
    
    # Bid and offer price bands for no risk hedging
    # offer_PB = [-1000]
    # bid_PB = [16000]
    
    # Define bid/offer price bands
    if system_assumptions["system_type"] == "BESS":
        offer_PB = offer_list[total_days_cumulative+day - (365*6+366*2)][1:11]
        bid_PB = bid_list[total_days_cumulative+day - (365*6+366*2)][1:11]
    else:
        offer_PB = offer_list[total_days_cumulative+day][1:11]
        bid_PB = bid_list[total_days_cumulative+day][1:11]
    
    offer_PB.reverse()

    # Run the optimisation solver to determine dispatch instructions
    dispatch_bidsOffers = schedulingModel(system_assumptions,linearisation_df, imperfectSP_day,day, offer_PB, bid_PB, current_state, phs_assumptions,forecasting_horizon)      
    
    # Run the bids and offers through the central dispatch model
    dispatchInstructions = dispatchModel(dispatch_bidsOffers,DP_day,SP_day,system_assumptions)
    
    # Send the dispatch instructions to the charging model
    chargingResults = chargingModel(current_state,system_assumptions,dispatchInstructions[0],day,phs_assumptions,year_count)
    current_state["SOC"] = chargingResults[0][-1] 
    current_state["Power"] = chargingResults[3]
    dispatchedCapacity = [chargingResults[1],chargingResults[2]]
    dispatchedEnergy = [chargingResults[4],chargingResults[5]]
    daily_cycles = chargingResults[6]
    
    # Determine settlement from actual charging behaviour 
    TA_day = settlementModel(system_assumptions,dispatchedEnergy,SP_day)
        
    return [dispatchedCapacity,TA_day,current_state,SP_day,daily_cycles]

def main(ifilename):
    '''
    Main function runs the arbitrage simulation over the system lifetime.

    Parameters
    ----------
    ifilename : string
        String of the filename for the job.

    Returns
    -------
    None.

    '''
    # Build system assumptions dictionary
    system_assumptions = {pd.read_csv("Assumptions/"+ifilename+"_ASSUMPTIONS.csv")['Assumption'][i]:pd.read_csv("Assumptions/"+ifilename+"_ASSUMPTIONS.csv")['Value'][i] for i in range(0,len(pd.read_csv("Assumptions/"+ifilename+"_ASSUMPTIONS.csv")['Assumption']))}
    
    # Define independent variables
    year = int(system_assumptions["Year"])
    region = system_assumptions["State"]             # NSW, QLD, SA, VIC, TAS
    forecasting_horizon = int(system_assumptions["forecasting_horizon"])  
    results_filename = "Results/"+ifilename+"_RESULTS.csv"
    
    # Build pumped hydro system assumption dictionary
    phs_assumptions = {index:{pd.read_csv("phs_assumptions.csv")['Parameter'][i]:pd.read_csv("phs_assumptions.csv")[index][i] for i in range(0,len(pd.read_csv("phs_assumptions.csv")['Parameter']))} for index in pd.read_csv("phs_assumptions.csv").columns if index != 'Parameter'}
    
    # Define linearisation parameters
    linearisation_df = pd.read_csv('linearisation.csv')
    
    # Define PHS system index max
    g_ind = int(system_assumptions["g_index_range"])
    h_ind = int(system_assumptions["h_index_range"])
    
    # Define system type
    system_type = system_assumptions["system_type"]
    
    # Establish the end-of-life variable
    EOL = int(system_assumptions["Lifetime"])
    
    # Define number of iterations
    if system_assumptions["system_type"] == "BESS":
        iteration_number = EOL
    else:
        iteration_number = 1
    
    # Establish simulation memory
    EOL_TA_dis = []
    EOL_TA_ch = []
    EOL_DischargedEnergy = []
    EOL_ChargedEnergy = []
    EOL_capacityFactor = []
    EOL_averageCycleTime = []
    EOL_finalSOCmax = []
    EOL_finalRcell = []
    EOL_data = []
    
    # Run the daily simulation for each day      
    # Define T+0 spot prices for region
    SP_df = pd.read_csv('SpotPrices.csv')
    SP_List = list(SP_df['Regions '+region+' Trading Price ($/MWh)']) 
        
    # Define T+0 pre-dispatch prices for region
    SP_df = pd.read_csv("predispatchSpotPrices.csv")
    imperfectSP_List = list(SP_df['Regions '+region+' Trading Price ($/MWh)'])
    imperfectSP_List.extend(imperfectSP_List[0:(forecasting_horizon - 48)])
    
    # Define T+0 offers and bids
    if system_assumptions["system_type"] == "BESS":        ###### Only SA 2018 - 2020
        offer_df = pd.read_csv('hornsdaleGenOffers.csv')        
        bid_df = pd.read_csv('hornsdaleLoadBids.csv')
    else:                                                  ###### Only QLD 2010 - 2020
        offer_df = pd.read_csv('wivenhoeGenOffers.csv')        
        bid_df = pd.read_csv('wivenhoeLoadBids.csv')
    offer_list =  offer_df.values.tolist()
    bid_list = bid_df.values.tolist()
    
    # Define initial state of system
    current_state = {"SOC":float(system_assumptions["SOC_initial"]),
                     "Power":float(system_assumptions["P_initial"]),
                     "SOC_max":float(system_assumptions["SOC_max_initial"]),
                     "Q_p_list_previous":g_ind*[0],
                     "Q_t_list_previous":h_ind*[0],
                     "P_p_list_previous":g_ind*[0],
                     "P_t_list_previous":h_ind*[0],
                     "R_cell":float(system_assumptions["R_cell_initial"]),
                     "cycLossCurrentSum":0,
                     "cycLossIntervals":0,
                     "calLossCurrentSum":0,
                     "calLossIntervals":0,
                     "Ah_throughput":0,
                     "calLossTime":0,
                     "SOC_max_loss_cal":0,
                     "SOC_max_loss_cyc":0,
                     "SOC_sum": 0.5,
                     "dispatch_intervals":0,
                     "cycle_tracker":0}
    
    for iteration in range(0,iteration_number):
        # Create simulation memory blocks
        annualDischargedEnergy = []
        annualChargedEnergy = []
        annual_TA_dis = []
        annual_TA_ch = []
        annual_SP = []
        annual_dailyCycles = []
        
        # Define T+0 dispatch prices for region
        DP_df = pd.read_csv('DispatchPrices_'+str(year)+'.csv')
        DP_List = list(DP_df['Regions '+region+' Dispatch Price ($/MWh)'])
        
        # Define number of days in year
        if year % 4 == 0:
            total_days = 366
        else:
                total_days = 365
            
        # Define cumulative days since 1 January 2010 04:30 until 1 January YEAR 04:30
        total_days_cumulative = (year-2010)*365+(year-2010+1)//4
            
        for day in range(0,total_days):
            print(ifilename, year, iteration, day)
            
            dailyOutputs = dailySimulation(system_assumptions,linearisation_df,SP_List,DP_List,day,year,current_state,total_days_cumulative,phs_assumptions,iteration,imperfectSP_List,forecasting_horizon,offer_list,bid_list)
            current_state = dailyOutputs[2]
            
            if system_type == "PHS":
                annualDischargedEnergy.append(sum([sum(dailyOutputs[0][0][t]) for t in range(0,288)])*(5/60))
                annualChargedEnergy.append(sum([sum(dailyOutputs[0][1][t]) for t in range(0,288)])*(5/60))
            else:
                annualDischargedEnergy.append(sum(dailyOutputs[0][0])*(5/60))
                annualChargedEnergy.append(sum(dailyOutputs[0][1])*(5/60))
            
            annual_TA_dis.append(sum(dailyOutputs[1][0]))
            annual_TA_ch.append(-sum(dailyOutputs[1][1]))
            annual_SP.extend(dailyOutputs[3])
            annual_dailyCycles.append(dailyOutputs[4])
                
        # Determine end of year results for systems with no degradation, assuming same discharging each year
        EOL_TA_dis.append(sum(annual_TA_dis))
        EOL_TA_ch.append(sum(annual_TA_ch))
        EOL_DischargedEnergy.append(sum(annualDischargedEnergy))
        EOL_ChargedEnergy.append(sum(annualChargedEnergy))
        EOL_capacityFactor.append(sum(annualDischargedEnergy) / (int(system_assumptions["power_capacity"]) * total_days * 24))
        EOL_averageCycleTime.append(sum(annual_dailyCycles) / total_days)
        EOL_finalSOCmax.append(current_state["SOC_max"])
        EOL_finalRcell.append(current_state["R_cell"])
            
        EOL_data.append([region,year,iteration+1,EOL_TA_dis[-1],EOL_TA_ch[-1],EOL_DischargedEnergy[-1],EOL_ChargedEnergy[-1],EOL_averageCycleTime[-1],EOL_capacityFactor[-1],EOL_finalSOCmax[-1],EOL_finalRcell[-1],"NA","NA","NA",forecasting_horizon,system_type,EOL])
            
        EOL_results = pd.DataFrame(data = EOL_data, columns=['Region','Year','Iteration','TA_discharging [$]','TA_charging [$]','DischargedEnergy [MWh]', 'ChargedEnergy [MWh]','averageCycleTime [cycles/day]','capacityFactor','final_SOCmax','final_RCell [Ohms]','RADP [$/MWh]','AADP [$/MWh]','Price Volatility','forecast_horizon','system type','lifetime'])
        EOL_results.to_csv(results_filename)
            
        if system_assumptions["system_type"] != "BESS":
            LCOS = EOL_LCOS(annualDischargedEnergy,annualChargedEnergy,annual_TA_dis,annual_TA_ch,system_assumptions,year)
            RADP = LCOS[0]
            AADP = LCOS[1]
            price_vol = volatility(annual_SP)
            EOL_data.append([region,year,"EOL",EOL*EOL_TA_dis[-1],EOL*EOL_TA_ch[-1],EOL*EOL_DischargedEnergy[-1],EOL*EOL_ChargedEnergy[-1],EOL_averageCycleTime[-1],EOL_capacityFactor[-1],EOL_finalSOCmax[-1],EOL_finalRcell[-1],RADP,AADP,price_vol,forecasting_horizon,system_type,EOL])
            EOL_results = pd.DataFrame(data = EOL_data, columns=['Region','Year','Iteration','TA_discharging [$]','TA_charging [$]','DischargedEnergy [MWh]', 'ChargedEnergy [MWh]','averageCycleTime [cycles/day]','capacityFactor','final_SOCmax','final_RCell [Ohms]','RADP [$/MWh]','AADP [$/MWh]','Price Volatility','forecast_horizon','system type','lifetime'])
            EOL_results.to_csv(results_filename)
        
    # Determine EOL resultsfor BESS
    if system_assumptions["system_type"] == "BESS":
        LCOS = EOL_LCOS_Deg(EOL_DischargedEnergy,EOL_ChargedEnergy,EOL_TA_dis,EOL_TA_ch,system_assumptions,year,EOL)
        RADP = LCOS[0]
        AADP = LCOS[1]
        price_vol = volatility(annual_SP)
        EOL_data.append([region,year,"EOL",sum(EOL_TA_dis),sum(EOL_TA_ch),sum(EOL_DischargedEnergy),sum(EOL_ChargedEnergy),np.average(EOL_averageCycleTime),sum(EOL_DischargedEnergy) / (int(system_assumptions["power_capacity"]) * EOL * total_days * 24),EOL_finalSOCmax[-1],EOL_finalRcell[-1],RADP,AADP,price_vol,forecasting_horizon,system_type,EOL])
        EOL_results = pd.DataFrame(data = EOL_data, columns=['Region','Year','Iteration','TA_discharging [$]','TA_charging [$]','DischargedEnergy [MWh]', 'ChargedEnergy [MWh]','averageCycleTime [cycles/day]','capacityFactor','final_SOCmax','final_RCell [Ohms]','RADP [$/MWh]','AADP [$/MWh]','Price Volatility','forecast_horizon','system type','lifetime'])
        EOL_results.to_csv(results_filename) 

if __name__ == '__main__':
    # Create dataframe of filenames
    result_filenames = pd.read_csv("result_filenames.csv")["Filename"].values.tolist()
    
    # Define the number of workers to be spawned
    try:
        workersCount = len(os.sched_getaffinity(0))
    except:
        workersCount = os.cpu_count()

    # Spawn workers and run the embarassingly parallel processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=workersCount) as executor:
        results = executor.map(main, result_filenames)
