########    REMOVE THESE IMPORTS IF THEY ARE NOT SPECIFICALLY RELEVANT TO MAIN. MOVE TO RELEVANT MODULES

# Import external libraries
import pandas as pd
import numpy as np
import concurrent.futures
import os

# Import classes
import market
import memory
import participant
import storage_system
import battery
import phs
import general_systems

# Import functional modules
import scheduling
import dispatch
import charging
import settlement
import lcos

# Import results modules
import volatility

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
    dispatch_bidsOffers = scheduling.schedulingModel(system_assumptions,linearisation_df, imperfectSP_day,day, offer_PB, bid_PB, current_state, phs_assumptions,forecasting_horizon)      
    
    # Run the bids and offers through the central dispatch model
    dispatchInstructions = dispatch.dispatchModel(dispatch_bidsOffers,DP_day,SP_day,system_assumptions)
    
    # Send the dispatch instructions to the charging model
    chargingResults = charging.chargingModel(current_state,system_assumptions,dispatchInstructions[0],day,phs_assumptions,year_count)
    current_state["SOC"] = chargingResults[0][-1] 
    current_state["Power"] = chargingResults[3]
    dispatchedCapacity = [chargingResults[1],chargingResults[2]]
    dispatchedEnergy = [chargingResults[4],chargingResults[5]]
    daily_cycles = chargingResults[6]
    
    # Determine settlement from actual charging behaviour 
    TA_day = settlement.settlementModel(system_assumptions,dispatchedEnergy,SP_day)
        
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
    
    # Build pumped hydro system assumption dictionary
    phs_assumptions = {index:{pd.read_csv("phs_assumptions.csv")['Parameter'][i]:pd.read_csv("phs_assumptions.csv")[index][i] for i in range(0,len(pd.read_csv("phs_assumptions.csv")['Parameter']))} for index in pd.read_csv("phs_assumptions.csv").columns if index != 'Parameter'}
    
    # Define linearisation parameters
    linearisation_df = pd.read_csv('linearisation.csv')

    # Define independent variables
    year = int(system_assumptions["Year"])
    region = system_assumptions["State"]             # NSW, QLD, SA, VIC, TAS
    forecasting_horizon = int(system_assumptions["forecasting_horizon"])  
    results_filename = "Results/"+ifilename+"_RESULTS.csv"
    
    # Create the storage system object
    if system_assumptions["system_type"] == "BESS":
        storage_system_type_inst = battery(system_assumptions)
    elif system_assumptions["system_type"] == "PHS":
        storage_system_type_inst = phs(system_assumptions, phs_assumptions)
    
    storage_system_gen_inst = general_systems(system_assumptions, linearisation_df)

    storage_system_inst = storage_system(storage_system_type_inst, storage_system_gen_inst)
    
    # Define number of iterations
    if storage_system_inst.type == "BESS":
        iteration_number = storage_system_inst.lifetime
    else:
        iteration_number = 1
    
    # Establish simulation memory
    simulation_memory = memory()
    
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
