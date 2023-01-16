'''
Define the classes relevant to pumped hydro systems.

Classes
-------
pump_turb

phs
'''

import numpy as np
import constants as const
import debug

class pump_turb:
    '''
    Class that defines pump or turbine objects.
    '''
    def __init__(self, unit_type, phs_assumptions, index):
        if unit_type == "pump":
            index_pref = "g"
        elif unit_type == "turbine":
            index_pref = "h"
        else:
            print("Unknown pumped hydro component type")
        
        # Constant parameters initialised
        self.type = unit_type # Whether the object is a pump or a turbine
        self.id = index_pref + str(index)
        self.P_rated = int(phs_assumptions[self.id]["P_rated [MW]"]) # Rated power of the pump/turbine [MW]
        self.Q_peak = int(phs_assumptions[self.id]["Q_peak [m3/s]"]) # Flow rate of the pump/turbine at peak efficiency [m^3/s]
        self.efficiency_peak = float(phs_assumptions[self.id]["Efficiency_peak"]) # Peak efficiency of the pump/turbine (0,1]

        # Current state parameters initialised
        self.Q_previous = 0 # Flow rate from the previous interval [m^3/s]
        self.P_previous = 0 # Power from the previous interval [MW]
        self.Q_t = 0 # Flow rate of the current interval [m^3/s]
        self.P_t = 0 # Power of the current interval [MW]
        self.efficiency_t = 0.91 # Efficiency of the unit in the current dispatch interval
        self.efficiency_previous = 0.91 # Efficiency of the unit in the previous dispatch interval
        self.V_transient_adjust_t = 0 # Adjustment to reservoir volume based on the transient effects of ramping up/down [m^3]
        self.RT_t = 0 # Ramp time for the unit in the dispatch interval [s]

class phs:
    '''
    Input to the storage_system constructor for pumped hydro system objects.
    '''
    def __init__(self,assumptions, phs_assumptions):
        '''
        Initialise attributes of a battery object.

        Parameters
        ----------
        assumptions : dataframe
            Dataframe of assumptions defined by the user in the ASSUMPTIONS.csv file.
        phs_assumptions : dataframe
            Dataframe of assumptions defined by the user in the phs_assumptions.csv file.
        '''

        self.obj_type = "phs"

        # Reservoir parameters initialised        
        self.V_res_u = int(assumptions["V_res_upper"]) # Total volume of the upper reservoir [m^3]
        self.V_res_l = int(assumptions["V_res_lower"]) # Total volume of the lower reservoir [m^3]
        self.H_r = float(assumptions["H_r"]) #
        self.H_ur = float(assumptions["H_ur"]) #
        self.H_lr = float(assumptions["H_lr"]) #

        # Pump parameters initialised
        self.g_range = int(assumptions["g_index_range"]) # Number of pumps attached to the penstock
        self.H_p_effective = int(assumptions["H_p_effective"]) # Effective head of pumps used during scheduling [m]
        self.H_pl_effective = int(assumptions["H_pl_effective"]) # Effective head loss of pumps assumed during scheduling [m]
        self.pump_penstock_d = float(assumptions["pump_penstock_pipe_diameter"]) # Diameter of the pump penstock [m]
        self.pump_penstock_l = float(assumptions["pump_penstock_pipe_length"]) # Length of the pump penstock [m]
        self.pump_K_fittings = float(assumptions["pump_K_fittings"]) #
        self.pump_abs_roughness = float(assumptions["pump_abs_roughness"])*10**(-3) #

        # Turbine parameters initialised
        self.h_range = int(assumptions["h_index_range"]) # Number of turbines attached to the penstock
        self.H_t_effective = int(assumptions["H_t_effective"]) # Effective head of turbines used during scheduling [m]
        self.H_tl_effective = int(assumptions["H_tl_effective"]) # Effective head loss of turbines assumed during scheduling
        self.turbine_penstock_d = float(assumptions["turbine_penstock_pipe_diameter"]) # Diameter of the turbine penstock [m]
        self.turbine_penstock_l = float(assumptions["turbine_penstock_pipe_length"]) # Length of the turbine penstock [m]
        self.turbine_K_fittings = float(assumptions["turbine_K_fittings"]) #
        self.turbine_abs_roughness = float(assumptions["turbine_abs_roughness"])*10**(-3) #

        # Assumed operating parameters initialised
        self.mol_lr = int(assumptions["MOL_lr"]) # Minimum operating level of lower reservoir [m]
        self.mol_ur = int(assumptions["MOL_ur"]) # Minimum operating level of upper reservoir [m]
        self.fsl_lr = self.H_lr # Full supply level of lower reservoir [m]
        self.fsl_ur = self.H_r + self.H_ur + self.H_lr # Full supply level of upper reservoir [m]
        self.rt_t_tnl = int(assumptions["RT_T_TNL"]) # Ramp time to transition from discharging to turbine no load [s]
        self.rt_tnl_p = int(assumptions["RT_TNL_P"]) # Ramp time to transition from turbine no load to pumping [s]
        self.rt_p_tnl = int(assumptions["RT_P_TNL"]) # Ramp time to transition from pumping to turbine no load [s]
        self.rt_tnl_t =int(assumptions["RT_TNL_T"]) # Ramp time to transition from turbine no load to discharging [s]

        # Current state parameters initialised
        self.pumps = [pump_turb("pump", phs_assumptions, g) for g in range(1,self.g_range+1)] # List of pump objects
        self.turbines = [pump_turb("turbine", phs_assumptions, h) for h in range(1,self.h_range+1)] # List of turbine objects
        self.Q_pump_penstock_t = 0 # Current flow rate in the pump penstock [m^3/s]
        self.Q_turbine_penstock_t = 0 # Current flow rate in the turbine penstock [m^3/s]
        self.Q_pump_penstock_pre = 0 # Flow rate in the pump penstock in previous interval [m^3/s]
        self.Q_turbine_penstock_pre = 0 # Flow rate in the turbine penstock in previous interval [m^3/s]
        self.H_pl_t = 0 # Current pump head loss [m]
        self.H_tl_t = 0 # Current turbine head loss [m]
        self.H_p_t = 0 # Current pump head [m]
        self.H_t_t = 0 # Current turbine head [m]
        self.efficiency_total_t = 0 # Current efficiency of the pumped hydro system based on all pumps or turbines

    def pumpHeadLoss(self):
        '''
        Calculate the pump head loss [m] for the PHS.

        Parameters
        ----------
        None.

        Returns
        -------
        self.H_pl_t : float
            Pump head loss based on pump flow rate for current dispatch interval.

        Side-effects
        ------------
        None.
        '''
        
        # Define variables and calculate head loss
        v_p = self.Q_pump_penstock_t/(0.25*np.pi*(self.pump_penstock_d**2))
        Re_p = (const.rho*v_p*self.pump_penstock_d)/const.mu
        F_p = (1.8*np.log10(6.9/Re_p + ((self.pump_abs_roughness/self.pump_penstock_d)/3.7)**1.11))**(-2)
        K_pipe = (F_p*self.pump_penstock_l)/self.pump_penstock_d
        K_p = K_pipe + self.pump_K_fittings
        self.H_pl_t = K_p*(v_p**2)/(2*const.gravity) 

        return self.H_pl_t
        
    def turbineHeadLoss(self):
        '''
        Calculate the turbine head loss [m] for the PHS.

        Parameters
        ----------
        None.

        Returns
        -------
        self.H_tl_t : float
            Turbine head loss based on turbine flow rate for current dispatch interval.

        Side-effects
        ------------
        None.
        '''
        
        # Define variables and calculate head loss
        if self.Q_turbine_penstock_t != 0:
            v_t = self.Q_turbine_penstock_t/(0.25*np.pi*(self.turbine_penstock_d**2))
            Re_t = (const.rho*v_t*self.turbine_penstock_d)/const.mu
            F_t = (1.8*np.log10(6.9/Re_t + ((self.turbine_abs_roughness/self.turbine_penstock_d)/3.7)**1.11))**(-2)
            K_pipe = (F_t*self.turbine_penstock_l)/self.turbine_penstock_d
            K_t = K_pipe + self.turbine_K_fittings
            self.H_tl_t = K_t*(v_t**2)/(2*const.gravity)
            
        else:
            self.H_tl_t = 0

        return self.H_tl_t

    def pumpHead(self, SOC):
        '''
        Calculate the net head for the pumps.

        Parameters
        ----------
        SOC : float
            State of charge of the system.

        Returns
        -------
        self.H_p_t : float
            Net head for pump in dispatch interval

        Side-effects
        ------------
        Call the pumpHeadLoss method to update the pump head loss.
        '''

        if self.Q_pump_penstock_t > 0:
            self.pumpHeadLoss()
        else:
            self.H_pl_t = 0
        
        # Calculate pump head
        self.H_p_t = (SOC*(self.fsl_ur - self.mol_ur) + self.mol_ur - ((1 - SOC)*(self.V_res_u/self.V_res_l)*(self.fsl_lr - self.mol_lr)) - self.mol_lr) + self.H_pl_t

        return self.H_p_t

    def Q_pump(self, pump_index, SOC):
        '''
        Update flow rate for a pump unit.

        Parameters
        ----------
        pump_index : integer
            Index for the pump unit to which the flow rate corresponds.
        SOC : float
            State of charge of the system.

        Returns
        -------
        self.pumps[pump_index].Q_t : float
            Pump flow rate during dispatch interval.
        
        Side-effects
        ------------
        Call the pumpHead method to update the pump head.
        Update the Q_previous attribute of the pump.
        Update Q_t attribute of the pump.
        Update self.Q_pump_penstock_t.
        Call the pumpEfficiency method to update the efficiency of the pump.
        '''

        self.pumpHead(SOC)

        self.pumps[pump_index].Q_previous = self.pumps[pump_index].Q_t

        
        if self.H_p_t < 0:
            self.pumps[pump_index].Q_t = 0
            
        else:    
            self.pumps[pump_index].Q_t = (self.pumps[pump_index].P_t*self.pumps[pump_index].efficiency_t*10**6)/(self.H_p_t*const.rho*const.gravity)
        
        self.Q_pump_penstock_t = sum([self.pumps[g].Q_t for g in range(0,self.g_range)])

        self.pumpEfficiency(pump_index)

        return self.pumps[pump_index].Q_t

    def Q_turbine(self, turbine_index, SOC):
        '''
        Update flow rate for a turbine unit.

        Parameters
        ----------
        turbine_index : integer
            Index for the turbine unit to which the flow rate corresponds.
        SOC : float
            State of charge of the system.

        Returns
        -------
        self.turbines[turbine_index].Q_t : float
            Turbine flow rate during dispatch interval.
        
        Side-effects
        ------------
        Call the turbineHead method to update the turbine head.
        Update the Q_previous attribute of the turbine.
        Update Q_t of the turbine.
        Call the turbineEfficiency method to update the efficiency of the turbine.
        '''

        self.turbineHead(SOC)

        self.turbines[turbine_index].Q_previous = self.turbines[turbine_index].Q_t
        
        if (self.turbines[turbine_index].efficiency_t == 0) or (self.H_t_t < 0):
            self.turbines[turbine_index].Q_t = 0
        
        else:
            self.turbines[turbine_index].Q_t = (self.turbines[turbine_index].P_t*10**6)/(self.H_t_t*const.rho*const.gravity*self.turbines[turbine_index].efficiency_t)
            
        self.turbineEfficiency(turbine_index)
        
        return self.turbines[turbine_index].Q_t

    def turbineHead(self, SOC):
        '''
        Calculate net head for the turbines.

        Parameters
        ----------
        SOC : float
            State of charge of the system.

        Returns
        -------
        self.H_t_t : float
            Net head of the turbines [m].

        Side-effects
        ------------
        Call the turbineHeadLoss method to update the turbine head loss.
        '''
        
        if self.Q_turbine_penstock_t > 0:
            self.turbineHeadLoss()
        else:
            self.H_tl_t = 0

        # Calculate turbine head
        self.H_t_t = (SOC*(self.fsl_ur - self.mol_ur) + self.mol_ur - ((1 - SOC)*(self.V_res_u/self.V_res_l)*(self.fsl_lr - self.mol_lr)) - self.mol_lr) - self.H_tl_t
        
        return self.H_t_t

    def pumpEfficiency(self, pump_index):
        '''
        Calculate the pump efficiency.

        Parameters
        ----------
        pump_index : integer
            Index for the pump unit to which the flow rate corresponds.

        Returns
        -------
        self.pumps[pump_index].efficiency_t : float
            Efficiency of the pump unit.

        '''
        self.pumps[pump_index].efficiency_t = 0.91
        return self.pumps[pump_index].efficiency_t

    def turbineEfficiency(self,turbine_index):
        '''
        Calculate the turbine efficiency based on empirically fitted model.

        Parameters
        ----------
        turbine_index : integer
            Index 'h' of the turbine.

        Returns
        -------
        self.turbines[turbine_index].efficiency_t : float
            Efficiency of the turbine unit.

        Side-effects
        ------------
        None.
        '''

        # Define parameters
        Q_ratio_turb_h = self.turbines[turbine_index].Q_t / self.turbines[turbine_index].Q_peak
        
        a = [-0.1421,3.044,-3.6718,2.3929,-0.7062]
        
        # Calculate turbine efficiency
        if self.turbines[turbine_index].Q_t > 0.01:
            self.turbines[turbine_index].efficiency_t = sum(a[n]*(Q_ratio_turb_h**n) for n in range(0,len(a)))
        else:
            self.turbines[turbine_index].efficiency_t = 0
        return self.turbines[turbine_index].efficiency_t
    
    def testToCurrent(self):
        '''
        Update the values for the previous dispatch interval if the tested charging/discharging was accepted.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        Side-effects
        ------------
        Updates the Q_previous attributes of all pumps and turbines.
        Updates self.Q_pump_penstock_pre.
        Updates self.Q_turbine_penstock_pre.
        '''

        for g in range(0,self.g_range):
            self.pumps[g].Q_previous = self.pumps[g].Q_t
            self.pumps[g].P_previous = self.pumps[g].P_t

        for h in range(0, self.h_range):
            self.turbines[h].Q_previous = self.turbines[h].Q_t
            self.turbines[h].P_previous = self.turbines[h].P_t

        self.Q_pump_penstock_pre = self.Q_pump_penstock_t

        self.Q_turbine_penstock_pre = self.Q_turbine_penstock_t

    def idleInterval(self):
        '''
        Update the values for the previous dispatch interval if the tested charging/discharging was rejected.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        Side-effects
        ------------
        Updates the Q_previous attributes of all pumps and turbines.
        Updates self.Q_pump_penstock_pre.
        Updates self.Q_turbine_penstock_pre.
        '''

        for g in range(0,self.g_range):
            self.pumps[g].Q_previous = 0
            self.pumps[g].P_previous = 0

        for h in range(0, self.h_range):
            self.turbines[h].Q_previous = 0
            self.turbines[h].P_previous = 0

        self.Q_pump_penstock_pre = 0

        self.Q_turbine_penstock_pre = 0

    def steadyStateCalc(self, dispatch_instruction_powers, SOC_pre, delT):
        '''
        Determine the steady state flow of the system in the dispatch interval.

        Parameters
        ----------
        dispatch_instruction_powers : list
            List of floats defining the charging and discharging instructions for each pump and turbine.
        SOC_pre : float
            State-of-charge of the system in the previous dispatch interval.
        delT : float
            Dispatch interval length.

        Returns
        -------
        SOC_exp : float
            The tested state-of-charge assuming that the storage system follows the dispatch instructions.

        Side-effects
        ------------
        Update self.Q_pump_penstock_t
        Update self.Q_turbine_penstock_t
        Update self.H_pl_t
        Update the efficiency_t attribute of all pumps and turbines.
        Update the P_t attribute of all pumps and turbines.
        Call the Q_pump function to update the pump flow rate
        Call the pumpHead function to update the pump head
        Update self.H_tl_t
        Call the Q_turbine function to update the turbine flow rate
        Call the turbineHead function to update the turbine head
        Update self.efficiency_total_t
        '''

        # Initialise parameters
        H_pl_initial = 0
        H_tl_initial = 0
        SOC_exp = SOC_pre
            
        # Charging behaviour
        if (sum(dispatch_instruction_powers) < 0):
            self.Q_pump_penstock_t = 0
            self.Q_turbine_penstock_t = 0
            self.H_pl_t = -1

            for g in range(0,self.g_range):
                self.pumps[g].efficiency_t = 0.91
                
            # Search for steady-state pump variable values
            while abs(self.H_pl_t - H_pl_initial) > 0.01*self.H_pl_t:
                H_pl_initial = self.H_pl_t
                # Calculate pump flow rates
                for g in range(0,self.g_range):
                    self.pumps[g].P_t = -dispatch_instruction_powers[g]
                    self.Q_pump(g, SOC_exp)
                # Send pump flows to pump penstock
                self.Q_pump_penstock_t = sum([self.pumps[g].Q_t for g in range (0,self.g_range)])
                self.pumpHead(SOC_exp)
                
            # Calculate new SOC
            SOC_exp = ((delT*3600)/self.V_res_u)*(self.Q_pump_penstock_t - self.Q_turbine_penstock_t)+SOC_pre
                
        elif (sum(dispatch_instruction_powers) > 1):                
            self.Q_pump_penstock_t = 0
            self.Q_turbine_penstock_t = 0
            self.H_tl_t = -1

            for h in range(0,self.h_range):
                self.turbines[h].efficiency_t = 0.91
                
            # Search for steady-state turbine variable values
            while abs(self.H_tl_t - H_tl_initial) > 0.01*self.H_tl_t:
                H_tl_initial = self.H_tl_t
                # Calculate turbine flow rates
                for h in range(0,self.h_range):  
                    self.turbines[h].P_t = dispatch_instruction_powers[h]
                    self.Q_turbine(h, SOC_exp)
                        
                # Send turbine flows to turbine penstock  
                self.Q_turbine_penstock_t = sum([self.turbines[h].Q_t for h in range(0,self.h_range)])
                self.turbineHead(SOC_exp)
                    
                self.efficiency_total_t = 0
                    
                # Calculate the overall turbine efficiency
                for h in range(0,self.h_range):
                    effProp = (self.turbines[h].Q_t / self.Q_turbine_penstock_t)*self.turbines[h].efficiency_t
                    self.efficiency_total_t += effProp
                
            # Update discharging variables for the dispatch interval
            SOC_exp = ((delT*3600)/self.V_res_u)*(self.Q_pump_penstock_t - self.Q_turbine_penstock_t)+SOC_pre
                                
        else:
            # Update variables if system is idleing
            for g in range(0,self.g_range):
                self.pumps[g].Q_t = 0
                
            for h in range(0,self.h_range):
                self.turbines[h].Q_t = 0

            SOC_exp = SOC_pre
        
        return SOC_exp
    
    def transientAdjust(self, SOC_exp):
        '''
        Determine the reservoir volume adjustment based on the transient ramp times between each state of pumping and generating.

        Parameters
        ----------
        SOC_exp : float
            The tested state-of-charge assuming that the storage system follows the dispatch instructions, no transient adjustment.

        Returns
        -------
        SOC_exp : float
            The tested state-of-charge assuming that the storage system follows the dispatch instructions, with transient adjustment.

        Side-effects
        ------------
        Update the RT_t attribute of all pumps and turbines.
        Update the V_transient_adjust_t attribute of all pumps and turbines.
        '''

        if (self.Q_turbine_penstock_t == self.Q_turbine_penstock_pre):
            
            for g in range(0,self.g_range):
                if self.Q_pump_penstock_t < self.Q_pump_penstock_pre:
                    self.pumps[g].RT_t = np.abs(self.pumps[g].Q_t - self.pumps[g].Q_previous) / self.pumps[g].Q_peak * self.rt_p_tnl
                else:
                    self.pumps[g].RT_t = np.abs(self.pumps[g].Q_t - self.pumps[g].Q_previous) / self.pumps[g].Q_peak * self.rt_tnl_p
                self.pumps[g].V_transient_adjust_t = self.pumps[g].RT_t*(self.pumps[g].Q_previous - self.pumps[g].Q_t)/2

            for h in range(0, self.h_range):
                self.turbines[h].RT_t = 0
                self.turbines[h].V_transient_adjust_t = 0
                
        elif (self.Q_pump_penstock_t == self.Q_pump_penstock_pre):
                
            for g in range(0,self.g_range):
                self.pumps[g].RT_t = 0
                self.pumps[g].V_transient_adjust_t = 0
                
            for h in range(0,self.h_range):
                if self.Q_turbine_penstock_t < self.Q_turbine_penstock_pre:
                    self.turbines[h].RT_t = np.abs(self.turbines[h].Q_previous - self.turbines[h].Q_t)/self.turbines[h].Q_peak * self.rt_t_tnl
                else: 
                    self.turbines[h].RT_t = np.abs(self.turbines[h].Q_previous - self.turbines[h].Q_t)/self.turbines[h].Q_peak * self.rt_tnl_t
                self.turbines[h].V_transient_adjust_t = self.turbines[h].RT_t*(self.turbines[h].Q_t - self.turbines[h].Q_previous)/2
                    
        elif (self.Q_pump_penstock_t < self.Q_pump_penstock_pre) and (self.Q_turbine_penstock_t > self.Q_turbine_penstock_pre):
                
            for g in range(0,self.g_range):
                self.pumps[g].RT_t = np.abs(self.Q_pump_penstock_t - self.Q_pump_penstock_pre)/self.pumps[g].Q_peak * self.rt_p_tnl
                self.pumps[g].V_transient_adjust_t = self.pumps[g].RT_t*(self.Q_pump_penstock_pre - self.Q_pump_penstock_t)/2
                    
            for h in range(0,self.h_range):
                self.turbines[h].RT_t = np.abs(self.Q_turbine_penstock_pre - self.Q_turbine_penstock_t)/self.turbines[h].Q_peak * self.rt_tnl_t
                self.turbines[h].V_transient_adjust_t = (self.turbines[h].RT_t/2 + max([self.pumps[g1].RT_t for g1 in range(0,self.g_range)]))*(self.Q_turbine_penstock_t - self.Q_turbine_penstock_pre)                     
                    
        elif (self.Q_pump_penstock_t > self.Q_pump_penstock_pre) and (self.Q_turbine_penstock_t < self.Q_turbine_penstock_pre):
                
            for h in range(0,self.h_range):
                self.turbines[h].RT_t = np.abs(self.Q_turbine_penstock_pre - self.Q_turbine_penstock_t)/self.turbines[h].Q_peak * self.rt_t_tnl
                self.turbines[h].V_transient_adjust_t = self.turbines[h].RT_t*(self.Q_turbine_penstock_t - self.Q_turbine_penstock_pre)/2
                
            for g in range(0,self.g_range):
                self.pumps[g].RT_t = np.abs(self.Q_pump_penstock_t - self.Q_pump_penstock_pre)/self.pumps[g].Q_peak * self.rt_tnl_p
                self.pumps[g].V_transient_adjust_t = (self.pumps[g].RT_t/2 + max([self.turbines[h1].RT_t for h1 in range(0,self.h_range)]))*(self.Q_pump_penstock_pre - self.Q_pump_penstock_t)
            
        else:
            for g in range(0,self.g_range):
                self.pumps[g].RT_t = 0
                self.pumps[g].V_transient_adjust_t = 0
            for h in range(0, self.h_range):
                self.turbines[h].RT_t = 0
                self.turbines[h].V_transient_adjust_t = 0
            
        # Update the SOC with the transient adjustments
        SOC_exp += (1/self.V_res_u)*(sum([self.pumps[g].V_transient_adjust_t for g in range(0,self.g_range)]) + sum([self.turbines[h].V_transient_adjust_t for h in range(0,self.h_range)]))

        return SOC_exp

            