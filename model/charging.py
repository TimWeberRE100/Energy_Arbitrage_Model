import memory
import debug
import display

def chargingModel(dispatchInstructions,day,year,storage_system_inst, market_inst, dp_list):
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
    dp_list : list
        List of dispatch prices for each dispatch interval in the trading day.

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
    memory.dispatch_prices = dp_list
    
    # Perform charging/discharging operation for each dispatch interval 
    for t in range(0,len(dispatchInstructions)):
        # Perform the battery charging operation
        if storage_system_inst.type == 'BESS':
            
            # Update the battery attributes
            storage_system_inst.incrementDispatchInterval()
            storage_system_inst.U_OCV_assign(storage_system_inst.SOC_current)
            storage_system_inst.eff_volt(storage_system_inst.SOC_current, storage_system_inst.P_current, storage_system_inst.energy_capacity)

            # Update the daily memory            
            daily_memory.U_batt_day.append(storage_system_inst.U_batt)
            daily_memory.eff_volt_day.append(storage_system_inst.efficiency_volt)
            
            # Create the test SOC update
            if dispatchInstructions[t] < 0:
                SOC_exp = storage_system_inst.SOC_pre - (1/storage_system_inst.energy_capacity)*storage_system_inst.efficiency_volt*storage_system_inst.efficiency_sys*dispatchInstructions[t]*delT
                daily_memory.behaviour.append(-1)
            elif dispatchInstructions[t] > 0:
                SOC_exp = storage_system_inst.SOC_pre - (1/storage_system_inst.energy_capacity)*(1/(storage_system_inst.efficiency_volt*storage_system_inst.efficiency_sys))*dispatchInstructions[t]*delT
                daily_memory.behaviour.append(1)
            else:
                SOC_exp = storage_system_inst.SOC_pre
                daily_memory.behaviour.append(0)

        # Perform the pumped hydro charging operation
        elif storage_system_inst.type == 'PHS':
            # Charging behaviour
            SOC_exp = storage_system_inst.steadyStateCalc(dispatchInstructions[t], storage_system_inst.SOC_pre, delT)

            if (sum(dispatchInstructions[t]) < 0):
                daily_memory.update_phs(-1, storage_system_inst.H_pl_t, storage_system_inst.H_p_t, storage_system_inst.Q_pump_penstock_t,0,0,0,0)
                       
            elif (sum(dispatchInstructions[t]) > 1):                
                daily_memory.update_phs(1,0,0,0,storage_system_inst.H_tl_t, storage_system_inst.H_t_t, storage_system_inst.Q_turbine_penstock_t,storage_system_inst.efficiency_total_t)
                                
            else:
                daily_memory.update_phs(0,0,0,0,0,0,0,0)

            # Perform transient adjustment based on ramp times
            SOC_exp = storage_system_inst.transientAdjust(SOC_exp)

        # Determine the actual SOC update
        if storage_system_inst.type == "PHS":
            if daily_memory.behaviour[t] == -1 and SOC_exp <= storage_system_inst.SOC_max:
                storage_system_inst.updateSOC_current(SOC_exp)
                storage_system_inst.updateCycleTracker(0)
                
                daily_memory.update_general(dispatchInstructions[t],\
                    [0]*storage_system_inst.h_range,\
                    -sum(dispatchInstructions[t])*delT + sum((storage_system_inst.pumps[g].RT_t/2 + max([storage_system_inst.turbines[h].RT_t for h in range(0,storage_system_inst.h_range)]))/3600*(-storage_system_inst.pumps[g].P_previous + dispatchInstructions[t][g]) for g in range(0,storage_system_inst.g_range)),\
                    sum((storage_system_inst.turbines[h].RT_t/2)/3600*(storage_system_inst.turbines[h].P_previous) for h in range(0,storage_system_inst.h_range)),\
                    storage_system_inst.SOC_current)

                
                storage_system_inst.updateP_current(sum(dispatchInstructions[t]))

                storage_system_inst.testToCurrent()

            elif daily_memory.behaviour[t] == 1 and SOC_exp >= storage_system_inst.SOC_min:
                storage_system_inst.updateSOC_current(SOC_exp)
                if storage_system_inst.cycle_tracker == 0:
                    daily_cycles += 1
                    storage_system_inst.updateCycleTracker(1)

                daily_memory.update_general([0]*storage_system_inst.g_range,\
                    dispatchInstructions[t],\
                    -sum((storage_system_inst.pumps[g].RT_t/2)/3600*(storage_system_inst.pumps[g].P_previous) for g in range(0,storage_system_inst.g_range)),\
                    sum(dispatchInstructions[t])*delT + sum((storage_system_inst.turbines[h].RT_t/2 + max([storage_system_inst.pumps[g].RT_t for g in range(0,storage_system_inst.g_range)]))/3600*(storage_system_inst.turbines[h].P_previous - dispatchInstructions[t][h]) for h in range(0,storage_system_inst.h_range)),\
                    storage_system_inst.SOC_current)
                
                storage_system_inst.updateP_current(sum(dispatchInstructions[t]))

                storage_system_inst.testToCurrent()

            else:
                daily_memory.update_general([0]*storage_system_inst.g_range,\
                    [0]*storage_system_inst.h_range,\
                    0,\
                    0,\
                    storage_system_inst.SOC_pre)
                
                storage_system_inst.updateP_current(0)

                storage_system_inst.idleInterval()
        
        else:
            if daily_memory.behaviour[t] == -1 and SOC_exp <= storage_system_inst.SOC_max:
                storage_system_inst.updateSOC_current(SOC_exp)
                storage_system_inst.updateCycleTracker(0)

                daily_memory.update_general(dispatchInstructions[t],\
                    0,\
                    -dispatchInstructions[t]*delT,\
                    0,\
                    storage_system_inst.SOC_current)

                storage_system_inst.updateP_current(dispatchInstructions[t])

                storage_system_inst.SOC_max_aged(delT, storage_system_inst.SOC_current, storage_system_inst.SOC_pre, storage_system_inst.P_current)

                storage_system_inst.testToCurrent()

            elif daily_memory.behaviour[t] == 1 and SOC_exp >= storage_system_inst.SOC_min:
                storage_system_inst.updateSOC_current(SOC_exp)
                if storage_system_inst.cycle_tracker == 0:
                    daily_cycles += 1
                    storage_system_inst.updateCycleTracker(1)

                daily_memory.update_general(0,\
                    dispatchInstructions[t],\
                    0,\
                    dispatchInstructions[t]*delT,\
                    storage_system_inst.SOC_current)

                storage_system_inst.updateP_current(dispatchInstructions[t])

                storage_system_inst.SOC_max_aged(delT, storage_system_inst.SOC_current, storage_system_inst.SOC_pre, storage_system_inst.P_current)

                storage_system_inst.testToCurrent()

            else:
                daily_memory.update_general(0,\
                    0,\
                    0,\
                    0,\
                    storage_system_inst.SOC_pre)

                storage_system_inst.updateP_current(0)

                storage_system_inst.updateSOC_current(storage_system_inst.SOC_pre)
                storage_system_inst.SOC_max_aged(delT, storage_system_inst.SOC_current, storage_system_inst.SOC_pre, storage_system_inst.P_current)

                storage_system_inst.idleInterval()
        
        if storage_system_inst.type == "BESS":
            # Efficiency fading
            storage_system_inst.R_cell_calc(year,day,t, storage_system_inst.SOC_pre)

            daily_memory.update_bess(storage_system_inst.SOC_max_loss_cal,\
                storage_system_inst.SOC_max_loss_cyc,\
                storage_system_inst.R_cell,\
                storage_system_inst.SOC_max)       
    
    # Remove the initial values from the list
    daily_memory.SOC_day.pop(0)
    daily_memory.chargingCapacity.pop(0)
    daily_memory.dischargingCapacity.pop(0)

    if day == display.test_day and display.display_arg:
        display.chargingOutputsDay(storage_system_inst, daily_memory)
    
    return daily_memory, daily_cycles, storage_system_inst