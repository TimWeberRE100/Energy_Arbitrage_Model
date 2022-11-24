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