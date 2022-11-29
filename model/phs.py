import numpy as np
import constants as const

class pump_turb:
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
        self.P_rated = int(phs_assumptions[id]["P_rated [MW]"]) # Rated power of the pump/turbine [MW]
        self.Q_peak = int(phs_assumptions[id]["Q_peak [m3/s]"]) # Flow rate of the pump/turbine at peak efficiency [m^3/s]
        self.efficiency_peak = float(phs_assumptions[id]["Efficiency_peak"]) # Peak efficiency of the pump/turbine (0,1]

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
    def __init__(self,assumptions, phs_assumptions):
        self.obj_type = "phs"

        # Reservoir parameters initialised        
        self.V_res_u = int(assumptions["V_res_upper"]) # Total volume of the upper reservoir [m^3]
        self.V_res_l = int(assumptions["V_res_lower"]) # Total volume of the lower reservoir [m^3]
        self.H_r = int(assumptions["H_r"]) #
        self.H_ur = int(assumptions["H_ur"]) #
        self.H_lr = int(assumptions["H_lr"]) #

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
        None.

        Returns
        -------
        self.H_p_t : float
            Net head for pump in dispatch interval

        '''

        self.pumpHeadLoss()
        
        # Calculate pump head
        self.H_p_t = (SOC*(self.fsl_ur - self.mol_ur) + self.mol_ur - ((1 - SOC)*(self.V_res_u/self.V_res_l)*(self.fsl_lr - self.mol_lr)) - self.mol_lr) + self.H_pl_t

        return self.H_p_t

    def Q_pump(self, pump_index):
        '''
        Flow rate for a pump unit.

        Parameters
        ----------
        pump_index : integer
            Index for the pump unit to which the flow rate corresponds.

        Returns
        -------
        self.pumps[pump_index].Q_t : float
            Pump flow rate during dispatch interval.

        '''

        self.pumpHead()

        self.pumps[pump_index].Q_previous = self.pumps[pump_index].Q_t

        
        if self.H_p_t < 0:
            self.pumps[pump_index].Q_t = 0
            
        else:    
            self.pumps[pump_index].Q_t = (self.pumps[pump_index].P_t*self.pumps[pump_index].efficiency_t*10**6)/(self.H_p_t*const.rho*const.gravity)
        
        self.Q_pump_penstock_t = sum([self.pumps[g].Q_t for g in range(1,self.g_range+1)])

        self.pumpEfficiency()

        return self.pumps[pump_index].Q_t

    def Q_turbine(self, turbine_index):
        '''
        Flow rate for a turbine unit

        Parameters
        ----------
        turbine_index : integer
            Index for the turbine unit to which the flow rate corresponds.

        Returns
        -------
        self.turbines[turbine_index].Q_t : float
            Turbine flow rate during dispatch interval.

        '''

        self.turbineHead()

        self.turbines[turbine_index].Q_previous = self.turbines[turbine_index].Q_t
        
        if (self.turbines[turbine_index].efficiency_t == 0) or (self.H_t_t < 0):
            self.turbines[turbine_index].Q_t = 0
        
        else:
            self.turbines[turbine_index].Q_t = (self.turbines[turbine_index].P_t*10**6)/(self.H_t_t*const.rho*const.gravity*self.turbines[turbine_index].efficiency_t)
            
        self.turbineEfficiency()
        
        return self.turbines[turbine_index].Q_t

    def turbineHead(self, SOC):
        '''
        Calculate net head for the turbines.

        Parameters
        ----------
        None.

        Returns
        -------
        self.H_t_t : float
            Net head of the turbines [m].

        '''
        
        H_tl = self.turbineHeadLoss()
        
        # Calculate turbine head
        self.H_t_t = (SOC*(self.fsl_ur - self.mol_ur) + self.mol_ur - ((1 - SOC)*(self.V_res_u/self.V_res_l)*(self.fsl_lr - self.mol_lr)) - self.mol_lr) - H_tl
        
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
        for g in range(0,self.g_range):
            self.pumps[g].Q_previous = self.pumps[g].Q_t
            self.pumps[g].P_previous = self.pumps[g].P_t

        for h in range(0, self.h_range):
            self.turbines[h].Q_previous = self.turbines[h].Q_t
            self.turbines[h].P_previous = self.turbines[h].P_t

        self.Q_pump_penstock_pre = self.Q_pump_penstock_t

        self.Q_turbine_penstock_pre = self.Q_turbine_penstock_t

    def idleInterval(self):
        for g in range(0,self.g_range):
            self.pumps[g].Q_previous = 0
            self.pumps[g].P_previous = 0

        for h in range(0, self.h_range):
            self.turbines[h].Q_previous = 0
            self.turbines[h].P_previous = 0

        self.Q_pump_penstock_pre = 0

        self.Q_turbine_penstock_pre = 0