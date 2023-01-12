import pyomo.environ as pyo
from pyomo.opt.results import SolverStatus
import numpy as np
import pandas as pd

import battery
import constants as const

import debug
import display

def Pmax_ij_aged(SOC,Ubatt_SOC,storage_system_inst):
    '''
    Calculates the maximum power limit for a particular state of charge, accounting for efficiency fade.

    Parameters
    ----------
    SOC : float
        State of charge of the system.
    Ubatt_SOC : float
        Open-circuit voltage of the battery at the specified SOC.
    storage_system_inst : storage_system
        Object containing storage system parameters and current state.

    Returns
    -------
    Pmax_current : float
        Maximum power limit ensuring efficiency losses > 80%.

    '''

    numerator = ((storage_system_inst.efficiency_sys**2)/4) - (0.8 - 0.5*storage_system_inst.efficiency_sys)**2
    denominator = storage_system_inst.efficiency_sys*((storage_system_inst.U_batt_nom/Ubatt_SOC)**2)*storage_system_inst.R_cell*storage_system_inst.cell_e*storage_system_inst.U_cell_nom
    coefficient = SOC*storage_system_inst.energy_capacity*storage_system_inst.U_cell_nom
    
    Pmax_current = coefficient*(numerator/denominator)

    return Pmax_current

def Ploss_m_aged(power,storage_system_inst):
    '''
    Calculates the power loss for a particular dispatch power, accounting for efficiency fade.

    Parameters
    ----------
    power : integer
        Dispatch power of the system [MW].
    storage_system_inst : storage_system
        Object containing storage system parameters and current state.

    Returns
    -------
    Ploss_current : float
        Power loss at the specified dispatch power.

    '''
    numerator = power*storage_system_inst.R_cell*storage_system_inst.cell_e*storage_system_inst.U_cell_nom
    denominator = storage_system_inst.efficiency_sys*0.5*storage_system_inst.energy_capacity*storage_system_inst.U_cell_nom
    radicand = 0.25 - numerator / denominator
    
    if radicand < 0:
        eff_volt = 0.5
    else:
        eff_volt = 0.5 + np.sqrt(radicand)
    
    Ploss_current = (1-eff_volt*storage_system_inst.efficiency_sys)*power/(eff_volt*storage_system_inst.efficiency_sys)
    
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

def schedulingModel(SP,day,offer_PB,bid_PB,forecasting_horizon, storage_system_inst, participant_inst, market_inst):
    '''
    Uses Pyomo library to define the model, then cbc (COIN-OR branch-and-cut) solver to optimise solution.
    Assumes dispatch capacities can be real numbers.

    Parameters
    ----------
    SP : list
        Forecast spot prices for the scheduling period.
    day : integer
        Count of trading day being scheduled in the year.
    offer_PB : list
        Ordered list of generator offer price bands for the day.
    bid_PB : list
        Ordered list of load bid price bands for the day.
    forecasting_horizon : integer
        Number of trading intervals that scheduling module optimises the MILP.
    storage_system_inst : storage_system
        Object containing storage system parameters and current state.
    participant_inst : participant
        Object containing market participant parameters.
    market_inst : market
        Object containing market parameters.

    Returns
    -------
    list
        List of bid and offer capacities, bid and offer price bands, and scheduled behaviour.

    '''

    # Create abstract model object
    model = pyo.AbstractModel()
        
    # Declare index parameters and ranges
    s = forecasting_horizon # 48
    model.s = pyo.RangeSet(1,s)
        
    # Define fixed parameters
    model.mlf_load = pyo.Param(initialize=storage_system_inst.mlf_load)
    model.mlf_gen = pyo.Param(initialize=storage_system_inst.mlf_gen)
    model.dlf_load = pyo.Param(initialize=storage_system_inst.dlf_load)
    model.dlf_gen = pyo.Param(initialize=storage_system_inst.dlf_gen)
    model.delT = pyo.Param(initialize=(6*market_inst.dispatch_t)/60)
    model.SOCmax = pyo.Param(initialize=storage_system_inst.SOC_max)
    model.Pmin = pyo.Param(initialize=storage_system_inst.P_min)
    model.Pmax = pyo.Param(initialize=storage_system_inst.P_max)
    model.Ce = pyo.Param(initialize=storage_system_inst.energy_capacity)
    model.SOCinitial = pyo.Param(initialize=storage_system_inst.SOC_current)
    model.VOMd = pyo.Param(initialize=storage_system_inst.VOMd)
    model.VOMc = pyo.Param(initialize=storage_system_inst.VOMc)
    
    # Define system specific parameters/variables
    if storage_system_inst.type == 'BESS':
        # Declare index parameters and ranges
        j = 9
        i = 9
        m = 8
        model.j = pyo.RangeSet(1,j) # 9
        model.i = pyo.RangeSet(1,i) # 9
        model.m = pyo.RangeSet(1,m) # 8
        
        # Declare other parameters
        model.SOCmin = pyo.Param(initialize=storage_system_inst.SOC_min)
        
        # Declare binary variables
        model.w = pyo.Var(model.s,within=pyo.Binary)
        model.zc = pyo.Var(model.s,model.j,within=pyo.Binary)
        model.zd = pyo.Var(model.s,model.i,within=pyo.Binary)
        model.zcLoss = pyo.Var(model.s,model.m,within=pyo.Binary)
        model.zdLoss = pyo.Var(model.s,model.m,within=pyo.Binary)
        model.k = pyo.Var(model.s,within=pyo.Binary)
        
        # Declare linearisation parameters
        a_df = storage_system_inst.a
        b_df = storage_system_inst.b
        c_df = storage_system_inst.c
        d_df = storage_system_inst.d

        # Define SOC list for linearisation based on a and b
        SOC_list = np.cumsum(b_df)
        
        # Define power list for linearisation based on c and d
        Power_list = np.cumsum(c_df)
        
        # Define Pmax at each SOC        
        Pmax_list = []        
        for ii in range(0,len(a_df)):
            Ubatt_SOC = storage_system_inst.series_cells*battery.U_OCV_calc(SOC_list[ii])
            Pmax_list.append(Pmax_ij_aged(SOC_list[ii],Ubatt_SOC,storage_system_inst) / storage_system_inst.P_max) 
        
        # Define Ploss at each power
        P_loss_list = []
        for mm in range(0,len(c_df)):
            P_loss_list.append(Ploss_m_aged(Power_list[mm],storage_system_inst)) 
        
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
        
    elif storage_system_inst.type == 'PHS':
        # Declare index parameters and ranges
        j = 9
        i = 9
        m = 8
        g = storage_system_inst.g_range
        h = storage_system_inst.h_range
        model.m = pyo.RangeSet(1,m) # 8
        model.g = pyo.RangeSet(1,g) # 2
        model.h = pyo.RangeSet(1,h) # 2
        
        # Declare binary variables
        model.w = pyo.Var(model.s,within=pyo.Binary)
        model.zdLoss = pyo.Var(model.s,model.m,model.h,within=pyo.Binary)
        model.zcLoss = pyo.Var(model.s,model.m,model.g,within=pyo.Binary)
        
        # Declare linearisation parameters
        c_df = storage_system_inst.c
        d_df = storage_system_inst.d
        sdLoss_df = storage_system_inst.sdLoss
        scLoss_df = storage_system_inst.scLoss
        
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
            return {g:storage_system_inst.pumps[g-1].Q_peak for g in model.g}
        model.QpPeak = pyo.Param(model.g,initialize=initialize_QpPeak,within=pyo.Any)
        
        def initialize_QtPeak(model,h):
            return {h:storage_system_inst.turbines[h-1].Q_peak for h in model.h}
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
        model.volume_reservoir = pyo.Param(initialize=storage_system_inst.V_res_u)
        model.rho = pyo.Param(initialize=const.rho) # kg/m^3
        model.gravity = pyo.Param(initialize=const.gravity) #m/s^2
        model.Hpeffective = pyo.Param(initialize=storage_system_inst.H_p_effective)
        model.Hteffective = pyo.Param(initialize=storage_system_inst.H_t_effective)
        model.SOCmin = pyo.Param(initialize=storage_system_inst.SOC_min)
        
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
    
    if storage_system_inst.type == 'BESS':
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
                return model.SOC[s] == storage_system_inst.SOC_initial
            else:
                return pyo.Constraint.Skip
        model.finalSOC = pyo.Constraint(model.s, rule=finalSOC_rule)
        
        # Objective: Maximise the arbitrage value for the day
        def arbitrageValue_rule(model):
            return sum(model.delT*model.SP[s][s]*(model.dlf_gen*model.mlf_gen*model.D[s] - model.dlf_load*model.mlf_load*model.C[s]) - (model.VOMd*model.Ed[s]) - (model.VOMc*model.Ec[s]) for s in model.s)
        model.arbitrageValue = pyo.Objective(rule=arbitrageValue_rule, sense = pyo.maximize)  
    
    elif storage_system_inst.type == 'PHS':
        
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
                return model.SOC[s] == storage_system_inst.SOC_initial
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
    solverpath_exe= pd.read_csv("cbc_path.csv")['Filepath'].iloc[0]
    
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
    ws = []
    
    for d in range(1,49):
        if storage_system_inst.type == "PHS":
            unit_h_subOffers = []
            unit_g_subBids = []

            # Optimisation outputs
            for h in instance.h:
                unit_h_subOffers.append(instance.D[d,h].value)  
                
            for g in instance.g:
                if instance.C[d,g].value > 0.1:
                    unit_g_subBids.append(-storage_system_inst.pumps[g-1].P_rated)
                else:
                    unit_g_subBids.append(0)
            
            # Optimisation outputs                
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
            
        dispatch_offers.append(max_lessThan(offer_PB,float(instance.SP[d][d]),participant_inst.risk_level))
        dispatch_bids.append(min_greaterThan(bid_PB,float(instance.SP[d][d]),participant_inst.risk_level))
        ws.append(instance.w[d].value)

    if (day == display.test_day and display.display_arg):
        display.schedulingOutputs(instance, storage_system_inst, dispatch_offers, dispatch_bids)
        
    return [dispatch_offer_cap,dispatch_bid_cap,dispatch_offers,dispatch_bids,ws,Total_dispatch_cap]