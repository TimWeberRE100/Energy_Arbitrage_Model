import memory
import constants as const

def chargingModel(dispatchInstructions,day,year,storage_system_inst, market_inst):
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
            storage_system_inst.SOC_pre = daily_memory.SOC_day[-1]
            storage_system_inst.U_OCV_assign()
            storage_system_inst.R_cell_calc()
            storage_system_inst.eff_volt()

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
            # Initialise parameters
            storage_system_inst.H_pl_t = 0
            storage_system_inst.H_tl_t = 0
            storage_system_inst.turbineHead()
            storage_system_inst.pumpHead()
            
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