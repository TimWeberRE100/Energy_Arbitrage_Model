import numpy as np

import memory
import constants as const
from battery import U_OCV_calc

def chargingModel(dispatchInstructions,day,year,storage_system_inst, market_inst):
    '''
    Perform the charging/discharging according to the dispatch instructions for the trading day.

    Parameters
    ----------
    dispatchInstructions : list
        List of outputs from the dispatch model.
    day : integer
        Count of days in the year on the current trading day.
    phs_assumptions : dictionary
        Dictionary of assumed parameters for the PHS turbines and pumps.
    year : integer
        Count of the year iterations on the current trading day.
    storage_system_inst : storage_system
        Object containing storage system parameters and current state.
    market_inst : market
        Object containing market parameters.

    Returns
    -------
    daily_memory : memory_daily
        Object containing the stored values for each dispatch interval within the day.
    daily_cycles : int
        Count of the number of charge/discharge cycles within the day.
    storage_system_inst : storage_system
        Object containing the storage system parameters and current state.
    '''
    
    # Define system level parameters
    daily_cycles = 0
    
    # Define charging parameters
    delT = market_inst.dispatch_t/60
    
    # Define memory for the day
    daily_memory = memory.memory_daily(storage_system_inst)
    
    # Perform charging/discharging operation for each dispatch interval 
    for t in range(0,len(dispatchInstructions)):
        # Perform the battery charging operation
        if storage_system_inst.type == 'BESS':
            
            # Update the battery attributes
            storage_system_inst.U_OCV_assign()
            storage_system_inst.R_cell_calc()
            storage_system_inst.eff_volt()

            # Update the daily memory            
            daily_memory.U_batt_day.append(storage_system_inst.U_batt)
            daily_memory.eff_volt_day.append(storage_system_inst.efficiency_volt)
            
            # Create the test SOC update
            if dispatchInstructions[t] < 0:
                storage_system_inst.SOC_current = storage_system_inst.SOC_pre - (1/storage_system_inst.energy_capacity)*storage_system_inst.efficiency_volt*storage_system_inst.efficiency_sys*dispatchInstructions[t]*delT
                daily_memory.behaviour.append(-1)
            elif dispatchInstructions[t] > 0:
                storage_system_inst.SOC_current = storage_system_inst.SOC_pre - (1/storage_system_inst.energy_capacity)*(1/(storage_system_inst.efficiency_volt*storage_system_inst.efficiency_sys))*dispatchInstructions[t]*delT
                daily_memory.behaviour.append(1)
            else:
                storage_system_inst.SOC_current = storage_system_inst.SOC_pre
                daily_memory.behaviour.append(0)
        
        # Perform the pumped hydro charging operation
        elif storage_system_inst.type == 'PHS':
            # Initialise parameters
            H_pl_initial = 0
            H_tl_initial = 0
            
            # Charging behaviour
            if (sum(dispatchInstructions[t]) < 0):
                storage_system_inst.Q_pump_penstock_t = 0
                storage_system_inst.Q_turbine_penstock_t = 0
                storage_system_inst.H_pl_t = -1

                for g in range(0,storage_system_inst.g_range):
                    storage_system_inst.pumps[h].efficiency_t = 0.91
                
                # Search for steady-state pump variable values
                while abs(storage_system_inst.H_pl_t - H_pl_initial) > 0.01*storage_system_inst.H_pl_t:
                    H_pl_initial = storage_system_inst.H_pl_t
                    # Calculate pump flow rates
                    for g in range(0,storage_system_inst.g_range):
                        storage_system_inst.pumps[g].P_t = -dispatchInstructions[t][g]
                        storage_system_inst.Q_pump(g)
                    # Send pump flows to pump penstock
                    storage_system_inst.Q_pump_penstock_t = sum([storage_system_inst.pumps[g].Q_t for g in range (0,storage_system_inst.g_range)])
                    storage_system_inst.pumpHead()
                
                # Calculate new SOC
                storage_system_inst.SOC_current = ((delT*3600)/storage_system_inst.V_res_u)*(storage_system_inst.Q_pump_penstock_t - storage_system_inst.Q_turbine_penstock_t)+storage_system_inst.SOC_pre
                
                # Append variables to lists
                daily_memory.update_phs(-1, storage_system_inst.H_pl_t, storage_system_inst.H_p_t, storage_system_inst.Q_pump_penstock_t,0,0,0,0)
                       
            elif (sum(dispatchInstructions[t]) > 1):                
                storage_system_inst.Q_pump_penstock_t = 0
                storage_system_inst.Q_turbine_penstock_t = 0
                storage_system_inst.H_tl_t = -1

                for h in range(0,storage_system_inst.h_range):
                    storage_system_inst.turbine[h].efficiency_t = 0.91
                
                # Search for steady-state turbine variable values
                while abs(storage_system_inst.H_tl_t - H_tl_initial) > 0.01*storage_system_inst.H_tl_t:
                    H_tl_initial = storage_system_inst.H_tl_t
                    # Calculate turbine flow rates
                    for h in range(0,storage_system_inst.h_range):  
                        storage_system_inst.turbine[h].P_t = dispatchInstructions[t][h]
                        storage_system_inst.Q_turbine(h)
                        
                    # Send turbine flows to turbine penstock  
                    storage_system_inst.Q_turbine_penstock_t = sum([storage_system_inst.turbine[h].Q_t for h in range(0,storage_system_inst.h_range)])
                    storage_system_inst.turbineHead()
                    
                    storage_system_inst.efficiency_total_t = 0
                    
                    # Calculate the overall turbine efficiency
                    for h in range(0,storage_system_inst.h_range):
                        effProp = (storage_system_inst.turbines[h].Q_t / storage_system_inst.Q_turbine_penstock_t)*storage_system_inst.turbines[h].efficiency_t
                        storage_system_inst.efficiency_total_t += effProp
                
                # Update discharging variables for the dispatch interval
                storage_system_inst.SOC_current = ((delT*3600)/storage_system_inst.V_res_u)*(storage_system_inst.Q_pump_penstock_t - storage_system_inst.Q_turbine_penstock_t)+storage_system_inst.SOC_pre
                
                daily_memory.update_phs(1,0,0,0,storage_system_inst.H_tl_t, storage_system_inst.H_t_t, storage_system_inst.Q_turbine_penstock_t,storage_system_inst.efficiency_total_t,0,0,0)
                                
            else:
                # Update variables if system is idleing
                for g in range(0,storage_system_inst.g_range):
                    storage_system_inst.pumps[g].Q_t = 0
                
                for h in range(0,storage_system_inst.h_range):
                    storage_system_inst.turbines[h].Q_t = 0

                storage_system_inst.SOC_current = storage_system_inst.SOC_pre

                daily_memory.update_phs(0,0,0,0,0,0,0,0)

            # Perform transient adjustment based on ramp times
            if (storage_system_inst.Q_turbine_penstock_t == storage_system_inst.Q_turbine_penstock_pre):
            
                for g in range(0,storage_system_inst.g_range):
                    if storage_system_inst.Q_pump_penstock_t < storage_system_inst.Q_pump_penstock_pre:
                        storage_system_inst.pumps[g].RT_t = np.abs(storage_system_inst.pumps[g].Q_t - storage_system_inst.pumps[g].Q_previous) / storage_system_inst.pumps[g].Q_peak * storage_system_inst.rt_p_tnl
                    else:
                        storage_system_inst.pumps[g].RT_t = np.abs(storage_system_inst.pumps[g].Q_t - storage_system_inst.pumps[g].Q_previous) / storage_system_inst.pumps[g].Q_peak * storage_system_inst.rt_tnl_p
                    storage_system_inst.pumps[g].V_transient_adjust_t = storage_system_inst.pumps[g].RT_t*(storage_system_inst.pumps[g].Q_previous - storage_system_inst.pumps[g].Q_t)/2

                for h in range(0, storage_system_inst.h_range):
                    storage_system_inst.turbines[h].RT_t = 0
                    storage_system_inst.turbines[h].V_transient_adjust_t = 0
                
            elif (storage_system_inst.Q_pump_penstock_t == storage_system_inst.Q_pump_penstock_pre):
                
                for g in range(0,storage_system_inst.g_range):
                    storage_system_inst.pumps[g].RT_t = 0
                    storage_system_inst.pumps[g].V_transient_adjust_t = 0
                
                for h in range(0,storage_system_inst.h_range):
                    if storage_system_inst.Q_turbine_penstock_t < storage_system_inst.Q_turbine_penstock_pre:
                        storage_system_inst.turbines[h].RT_t = np.abs(storage_system_inst.turbines[h].Q_previous - storage_system_inst.turbines[h].Q_t)/storage_system_inst.turbines[h].Q_peak * storage_system_inst.turbines[h].rt_t_tnl
                    else: 
                        storage_system_inst.turbines[h].RT_t = np.abs(storage_system_inst.turbines[h].Q_previous - storage_system_inst.turbines[h].Q_t)/storage_system_inst.turbines[h].Q_peak * storage_system_inst.turbines[h].rt_tnl_t
                    storage_system_inst.turbines[h].V_transient_adjust_t = storage_system_inst.turbines[h].RT_t*(storage_system_inst.turbines[h].Q_t - storage_system_inst.turbines[h].Q_previous)/2
                    
            elif (storage_system_inst.Q_pump_penstock_t < storage_system_inst.Q_pump_penstock_pre) and (storage_system_inst.Q_turbine_penstock_t > storage_system_inst.Q_turbine_penstock_pre):
                
                for g in range(0,storage_system_inst.g_range):
                    storage_system_inst.pumps[g].RT_t = np.abs(storage_system_inst.Q_pump_penstock_t - storage_system_inst.Q_pump_penstock_pre)/storage_system_inst.pumps[g].Q_peak * storage_system_inst.rt_p_tnl
                    storage_system_inst.pumps[g].V_transient_adjust_t = storage_system_inst.pumps[g].RT_t*(storage_system_inst.Q_pump_penstock_pre - storage_system_inst.Q_pump_penstock_t)/2
                    
                for h in range(0,storage_system_inst.h_range):
                    storage_system_inst.turbines[h].RT_t = np.abs(storage_system_inst.Q_turbine_penstock_pre - storage_system_inst.Q_turbine_penstock_t)/storage_system_inst.turbines[h].Q_peak * storage_system_inst.rt_tnl_t
                    storage_system_inst.turbines[h].V_transient_adjust_t = (storage_system_inst.turbines[h].RT_t/2 + max([storage_system_inst.pumps[g1].RT_t for g1 in range(0,storage_system_inst.g_range)]))*(storage_system_inst.Q_turbine_penstock_t - storage_system_inst.Q_turbine_penstock_pre)                     
                    
            elif (storage_system_inst.Q_pump_penstock_t > storage_system_inst.Q_pump_penstock_pre) and (storage_system_inst.Q_turbine_penstock_t < storage_system_inst.Q_turbine_penstock_pre):
                
                for h in range(0,storage_system_inst.h_range):
                    storage_system_inst.turbines[h].RT_t = np.abs(storage_system_inst.Q_turbine_penstock_pre - storage_system_inst.Q_turbine_penstock_t)/storage_system_inst.turbines[h].Q_peak * storage_system_inst.rt_t_tnl
                    storage_system_inst.turbines[h].V_transient_adjust_t = storage_system_inst.turbines[h].RT_t*(storage_system_inst.Q_turbine_penstock_t - storage_system_inst.Q_turbine_penstock_pre)/2
                
                for g in range(0,storage_system_inst.g_range):
                    storage_system_inst.pumps[g].RT_t = np.abs(storage_system_inst.Q_pump_penstock_t - storage_system_inst.Q_pump_penstock_pre)/storage_system_inst.pumps[g].Q_peak * storage_system_inst.rt_tnl_p
                    storage_system_inst.pumps[g].V_transient_adjust_t = (storage_system_inst.pumps[g].RT_t/2 + max([storage_system_inst.turbines[h1].RT_t for h1 in range(0,storage_system_inst.h_range)]))*(storage_system_inst.Q_pump_penstock_pre - storage_system_inst.Q_pump_penstock_t)
            
            else:
                for g in range(0,storage_system_inst.g_range):
                    storage_system_inst.pumps[g].RT_t = 0
                    storage_system_inst.pumps[g].V_transient_adjust_t = 0
                for h in range(0, storage_system_inst.h_range):
                    storage_system_inst.turbines[h].RT_t = 0
                    storage_system_inst.turbines[h].V_transient_adjust_t = 0
            
            # Update the SOC with the transient adjustments
            storage_system_inst.SOC_current += (1/storage_system_inst.V_res_u)*(sum([storage_system_inst.pumps[g].V_transient_adjust_t for g in range(0,storage_system_inst.g_range)]) + sum([storage_system_inst.turbines[h].V_transient_adjust_t for h in range(0,storage_system_inst.h_range)]))
            
        # Determine the actual SOC update
        if storage_system_inst.type == "PHS":
            if daily_memory.behaviour[t] == -1 and storage_system_inst.SOC_current <= storage_system_inst.SOC_max:
                storage_system_inst.cycle_tracker = 0
                
                daily_memory.update_general(dispatchInstructions[t],\
                    [0]*storage_system_inst.h_range,\
                    -sum(dispatchInstructions[t])*delT + sum((storage_system_inst.pumps[g].RT_t/2 + max([storage_system_inst.turbines[h].RT_t for h in range(0,storage_system_inst.h_range)]))/3600*(-storage_system_inst.pumps[g].P_previous + dispatchInstructions[t][g]) for g in range(0,storage_system_inst.g_range)),\
                    sum((storage_system_inst.turbines[h].RT_t/2)/3600*(storage_system_inst.turbines[h].P_previous) for h in range(0,storage_system_inst.h_range)),\
                    storage_system_inst.SOC_current)

                
                storage_system_inst.P_current = sum(dispatchInstructions[t])

                storage_system_inst.testToCurrent()

            elif daily_memory.behaviour[t] == 1 and storage_system_inst.SOC_current >= storage_system_inst.SOC_min:
                if storage_system_inst.cycle_tracker == 0:
                    daily_cycles += 1
                    storage_system_inst.cycle_tracker = 1

                daily_memory.update_general([0]*storage_system_inst.g_range,\
                    dispatchInstructions[t],\
                    -sum((storage_system_inst.pumps[g].RT_t/2)/3600*(storage_system_inst.pumps[g].P_previous) for g in range(0,storage_system_inst.g_range)),\
                    sum(dispatchInstructions[t])*delT + sum((storage_system_inst.turbines[h].RT_t/2 + max([storage_system_inst.pumps[g].RT_t for g in range(0,storage_system_inst.g_range)]))/3600*(storage_system_inst.turbines[h].P_previous - dispatchInstructions[t][h]) for h in range(0,storage_system_inst.h_range)),\
                    storage_system_inst.SOC_current)
                
                storage_system_inst.P_current = sum(dispatchInstructions[t])

                storage_system_inst.testToCurrent()

            else:
                daily_memory.update_general([0]*storage_system_inst.g_range,\
                    [0]*storage_system_inst.h_range,\
                    0,\
                    0,\
                    storage_system_inst.SOC_pre)
                
                storage_system_inst.P_current = 0

                storage_system_inst.idleInterval()
        
        else:
            if daily_memory.behaviour[t] == -1 and storage_system_inst.SOC_current <= storage_system_inst.SOC_max:
                cycle_tracker = 0

                daily_memory.update_general(dispatchInstructions[t],\
                    0,\
                    -dispatchInstructions[t]*delT,\
                    0,\
                    storage_system_inst.SOC_current)

                storage_system_inst.P_current = dispatchInstructions[t]

                storage_system_inst.SOC_max_aged()

                storage_system_inst.testToCurrent()

            elif daily_memory.behaviour[t] == 1 and storage_system_inst.SOC_current >= storage_system_inst.SOC_min:
                if cycle_tracker == 0:
                    daily_cycles += 1
                    cycle_tracker = 1

                daily_memory.update_general(0,\
                    dispatchInstructions[t],\
                    0,\
                    dispatchInstructions[t]*delT,\
                    storage_system_inst.SOC_current)

                storage_system_inst.P_current = dispatchInstructions[t]

                storage_system_inst.SOC_max_aged()

                storage_system_inst.testToCurrent()

            else:
                daily_memory.update_general(0,\
                    0,\
                    0,\
                    0,\
                    storage_system_inst.SOC_pre)

                storage_system_inst.P_current = 0

                storage_system_inst.SOC_current = storage_system_inst.SOC_pre
                storage_system_inst.SOC_max_aged()

                storage_system_inst.idleInterval()
        
        if storage_system_inst.type == "BESS":
            # Capacity fading
            storage_system_inst.SOC_sum += daily_memory.SOC_day[-1]
            storage_system_inst.dispatch_intervals += 1
            
            # Efficiency fading
            storage_system_inst.R_cell_calc(year,day,t)

            daily_memory.update_bess(storage_system_inst.SOC_max_loss_cal,\
                storage_system_inst.SOC_max_loss_cyc,\
                storage_system_inst.R_cell,\
                storage_system_inst.SOC_max)       
    
    # Remove the initial values from the list
    daily_memory.SOC_day.pop(0)
    daily_memory.chargingCapacity.pop(0)
    daily_memory.dischargingCapacity.pop(0)
    
    return daily_memory, daily_cycles, storage_system_inst