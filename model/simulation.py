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
import display
import debug

def dailySimulation(SP,DP,day,year,total_days_cumulative,year_count,imperfectSP,forecasting_horizon,storage_system_inst, participant_inst, market_inst):
    
    '''
    Simulate arbitrage over one trading day.

    Parameters
    ----------
    SP : list
        List of spot prices for all trading intervals in the year.
    DP : list
        List of dispatch prices for all trading intervals in the year.
    day : integer
        Count of days in the year at the current trading day.
    year : integer
        Year for which price data belongs.
    total_days_cumulative : integer
        Count of days in the simulation at the current trading day.
    year_count : integer
        Count of iterations of the year at the current trading day.
    imperfectSP : list
        List of imperfectly forecast spot prices for all trading intervals in the year.
    forecasting_horizon : integer
        Number of trading intervals which the scheduling model optimises at once.
    storage_system_inst : storage_system
        Object containing storage system parameters and current state.
    participant_inst : participant
        Object containing market participant parameters.
    market_inst : market
        Object containing market parameters.

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
    offer_PB = [-1000]
    bid_PB = [16000]
    
    # Define bid/offer price bands
    # if storage_system_inst.type == "BESS":
    #     offer_PB = participant_inst.offers[total_days_cumulative+day - (365*6+366*2)][1:11]
    #     bid_PB = participant_inst.bids[total_days_cumulative+day - (365*6+366*2)][1:11]
    # else:
    #     offer_PB = participant_inst.offers[total_days_cumulative+day][1:11]
    #     bid_PB = participant_inst.bids[total_days_cumulative+day][1:11]
    
    offer_PB.reverse()

    # Run the optimisation solver to determine dispatch instructions
    dispatch_bidsOffers = scheduling.schedulingModel(imperfectSP_day,day, offer_PB, bid_PB, forecasting_horizon, storage_system_inst, participant_inst, market_inst)      
    
    # Run the bids and offers through the central dispatch model
    dispatchInstructions = dispatch.dispatchModel(dispatch_bidsOffers,DP_day,storage_system_inst)
    
    # Send the dispatch instructions to the charging model
    chargingResults = charging.chargingModel(dispatchInstructions[0],day,year_count, storage_system_inst, market_inst, DP_day)
    dispatchedCapacity = [chargingResults[0].dischargingCapacity,chargingResults[0].chargingCapacity]
    dispatchedEnergy = [chargingResults[0].dischargedEnergy,chargingResults[0].chargedEnergy]
    daily_cycles = chargingResults[1]
    storage_system_inst = chargingResults[2]
    
    # Determine settlement from actual charging behaviour 
    TA_day = settlement.settlementModel(storage_system_inst,dispatchedEnergy,SP_day)
        
    return dispatchedCapacity, TA_day, SP_day, daily_cycles, storage_system_inst

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
    system_assumptions = {pd.read_csv("assumptions/"+ifilename+"_ASSUMPTIONS.csv")['Assumption'][i]:pd.read_csv("assumptions/"+ifilename+"_ASSUMPTIONS.csv")['Value'][i] for i in range(0,len(pd.read_csv("assumptions/"+ifilename+"_ASSUMPTIONS.csv")['Assumption']))}
    
    # Build pumped hydro system assumption dictionary
    phs_assumptions = {index:{pd.read_csv("assumptions/phs_assumptions.csv")['Parameter'][i]:pd.read_csv("assumptions/phs_assumptions.csv")[index][i] for i in range(0,len(pd.read_csv("assumptions/phs_assumptions.csv")['Parameter']))} for index in pd.read_csv("assumptions/phs_assumptions.csv").columns if index != 'Parameter'}
    
    # Define linearisation parameters
    linearisation_df = pd.read_csv('assumptions/linearisation.csv')

    # Define independent variables
    year = int(system_assumptions["Year"])
    region = system_assumptions["State"]             # NSW, QLD, SA, VIC, TAS
    forecasting_horizon = int(system_assumptions["forecasting_horizon"])  
    results_filename = "results/"+ifilename+"_RESULTS.csv"
    
    # Create the storage system object
    if system_assumptions["system_type"] == "BESS":
        storage_system_type_inst = battery.battery(system_assumptions)
    elif system_assumptions["system_type"] == "PHS":
        storage_system_type_inst = phs.phs(system_assumptions, phs_assumptions)
    
    storage_system_gen_inst = general_systems.general_systems(system_assumptions, linearisation_df)

    storage_system_inst = storage_system.storage_system(storage_system_type_inst, storage_system_gen_inst)
    
    # Create market participant object
    participant_inst = participant.participant(system_assumptions)

    # Create market object
    market_inst = market.market(system_assumptions)

    # Define number of iterations
    if storage_system_inst.type == "BESS":
        iteration_number = storage_system_inst.lifetime
    else:
        iteration_number = 1
    
    # Establish simulation memory
    simulation_memory = memory.memory()
    
    # Run the daily simulation for each day      
    # Define T+0 spot prices for region
    SP_df = pd.read_csv('data/SpotPrices.csv')
    SP_List = list(SP_df['Regions '+region+' Trading Price ($/MWh)']) 
        
    # Define T+0 pre-dispatch prices for region
    SP_df = pd.read_csv("data/predispatchSpotPrices.csv")
    imperfectSP_List = list(SP_df['Regions '+region+' Trading Price ($/MWh)'])
    imperfectSP_List.extend(imperfectSP_List[0:(forecasting_horizon - 48)])
    
    for iteration in range(0,iteration_number):
        # Create annual memory
        annual_memory = memory.memory()
        
        # Define T+0 dispatch prices for region
        DP_df = pd.read_csv('data/DispatchPrices_'+str(year)+'.csv')
        DP_List = list(DP_df['Regions '+region+' Dispatch Price ($/MWh)'])
        
        # Define number of days in year
        if year % 4 == 0:
            total_days = 366
        else:
            total_days = 365
            
        # Define cumulative days since 1 January 2010 04:30 until 1 January YEAR 04:30
        total_days_cumulative = (year-2010)*365+(year-2010+1)//4
            
        #for day in range(0,total_days):
        for day in range(0,15):
            print(ifilename, year, iteration, day)
            
            dailyOutputs = dailySimulation(SP_List,DP_List,day,year,total_days_cumulative,iteration,imperfectSP_List,forecasting_horizon,storage_system_inst, participant_inst, market_inst)
            
            if storage_system_inst.type == "PHS":
                annual_memory.DischargedEnergy.append(sum([sum(dailyOutputs[0][0][t]) for t in range(0,288)])*(5/60))
                annual_memory.ChargedEnergy.append(sum([sum(dailyOutputs[0][1][t]) for t in range(0,288)])*(5/60))
            else:
                annual_memory.DischargedEnergy.append(sum(dailyOutputs[0][0])*(5/60))
                annual_memory.ChargedEnergy.append(sum(dailyOutputs[0][1])*(5/60))
            
            annual_memory.TA_dis.append(sum(dailyOutputs[1][0]))
            annual_memory.TA_ch.append(-sum(dailyOutputs[1][1]))
            annual_memory.SP.extend(dailyOutputs[2])
            annual_memory.dailyCycles.append(dailyOutputs[3])
            storage_system_inst = dailyOutputs[4]
                
            if day == display.test_day and display.display_arg:
                display.chargingOutputsLifetime(storage_system_inst, annual_memory)
        
        # Determine end of year results for systems with no degradation, assuming same discharging each year
        simulation_memory.TA_dis.append(sum(annual_memory.TA_dis))
        simulation_memory.TA_ch.append(sum(annual_memory.TA_ch))
        simulation_memory.DischargedEnergy.append(sum(annual_memory.DischargedEnergy))
        simulation_memory.ChargedEnergy.append(sum(annual_memory.ChargedEnergy))
        simulation_memory.capacityFactor.append(sum(annual_memory.DischargedEnergy) / (int(system_assumptions["power_capacity"]) * total_days * 24))
        simulation_memory.averageCycleTime.append(sum(annual_memory.dailyCycles) / total_days)
        simulation_memory.finalSOCmax.append(storage_system_inst.SOC_max)

        if storage_system_inst.type == "BESS":
            simulation_memory.finalRcell.append(storage_system_inst.R_cell)
        else:
            simulation_memory.finalRcell.append(0)

            
        simulation_memory.data.append([region,year,iteration+1,simulation_memory.TA_dis[-1],simulation_memory.TA_ch[-1],simulation_memory.DischargedEnergy[-1],simulation_memory.ChargedEnergy[-1],simulation_memory.averageCycleTime[-1],simulation_memory.capacityFactor[-1],simulation_memory.finalSOCmax[-1],simulation_memory.finalRcell[-1],"NA","NA","NA",forecasting_horizon,storage_system_inst.type,storage_system_inst.lifetime])

        EOL_results = pd.DataFrame(data = simulation_memory.data, columns=['Region','Year','Iteration','TA_discharging [$]','TA_charging [$]','DischargedEnergy [MWh]', 'ChargedEnergy [MWh]','averageCycleTime [cycles/day]','capacityFactor','final_SOCmax','final_RCell [Ohms]','RADP [$/MWh]','AADP [$/MWh]','Price Volatility','forecast_horizon','system type','lifetime'])
        EOL_results.to_csv(results_filename)
            
        if system_assumptions["system_type"] != "BESS":
            LCOS = lcos.EOL_LCOS(annual_memory.DischargedEnergy,annual_memory.TA_dis,annual_memory.TA_ch, storage_system_inst, year)
            RADP = LCOS[0]
            AADP = LCOS[1]
            price_vol = volatility.volatility(annual_memory.SP)
            simulation_memory.data.append([region,year,"EOL",storage_system_inst.lifetime*simulation_memory.TA_dis[-1],storage_system_inst.lifetime*simulation_memory.TA_ch[-1],storage_system_inst.lifetime*simulation_memory.DischargedEnergy[-1],storage_system_inst.lifetime*simulation_memory.ChargedEnergy[-1],simulation_memory.averageCycleTime[-1],simulation_memory.capacityFactor[-1],simulation_memory.finalSOCmax[-1],simulation_memory.finalRcell[-1],RADP,AADP,price_vol,forecasting_horizon,storage_system_inst.type,storage_system_inst.lifetime])
            EOL_results = pd.DataFrame(data = simulation_memory.data, columns=['Region','Year','Iteration','TA_discharging [$]','TA_charging [$]','DischargedEnergy [MWh]', 'ChargedEnergy [MWh]','averageCycleTime [cycles/day]','capacityFactor','final_SOCmax','final_RCell [Ohms]','RADP [$/MWh]','AADP [$/MWh]','Price Volatility','forecast_horizon','system type','lifetime'])
            EOL_results.to_csv(results_filename)
        
    # Determine EOL resultsfor BESS
    if system_assumptions["system_type"] == "BESS":
        LCOS = lcos.EOL_LCOS_Deg(simulation_memory.DischargedEnergy,simulation_memory.TA_dis,simulation_memory.TA_ch,year,storage_system_inst)
        RADP = LCOS[0]
        AADP = LCOS[1]
        price_vol = volatility.volatility(annual_memory.SP)
        simulation_memory.data.append([region,year,"EOL",sum(simulation_memory.TA_dis),sum(simulation_memory.TA_ch),sum(simulation_memory.DischargedEnergy),sum(simulation_memory.ChargedEnergy),np.average(simulation_memory.averageCycleTime),sum(simulation_memory.DischargedEnergy) / (storage_system_inst.power_capacity * storage_system_inst.lifetime * total_days * 24),simulation_memory.finalSOCmax[-1],simulation_memory.finalRcell[-1],RADP,AADP,price_vol,forecasting_horizon,storage_system_inst.type,storage_system_inst.lifetime])
        EOL_results = pd.DataFrame(data = simulation_memory.data, columns=['Region','Year','Iteration','TA_discharging [$]','TA_charging [$]','DischargedEnergy [MWh]', 'ChargedEnergy [MWh]','averageCycleTime [cycles/day]','capacityFactor','final_SOCmax','final_RCell [Ohms]','RADP [$/MWh]','AADP [$/MWh]','Price Volatility','forecast_horizon','system type','lifetime'])
        EOL_results.to_csv(results_filename) 
    
    if display.display_arg:
        display.chargingOutputsLifetime(storage_system_inst, simulation_memory)

if __name__ == '__main__':
    '''
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
    '''
    
    main("test")
